"""
MARC: Universal Multi-Agent Trace Parser
Framework-agnostic parser that normalizes execution logs from AG2/AutoGen,
MetaGPT, ChatDev, Magentic-One, and other MAS into clean turn sequences.

Author: Viruna
Project: MARC (Multi-Agent Realignment and Correction)
"""

import re
import ast
import json
from typing import List, Dict, Tuple, Optional, Any

class UniversalTraceParser:
    """
    Parses heterogeneous multi-agent logs into standardized turn objects:
    (turn_id, speaker_name, message_content, is_goal)
    """

    def __init__(self):
        # Regex patterns for various agent frameworks
        self.role_header_patterns = [
            # ChatDev format: [Speaker]: content or Phase: ... [Speaker]: content
            re.compile(r'\[(?P<speaker>[A-Za-z0-9_\-\s]{2,30})\]\s*:\s*(?P<content>.*?)(?=\n\[[A-Za-z0-9_\-\s]{2,30}\]\s*:|\Z)', re.DOTALL),
            # MetaGPT format: ## Role: Speaker \n content
            re.compile(r'##\s*(?:Role|Speaker|Agent)?\s*:?\s*(?P<speaker>[A-Za-z0-9_\-\s]{2,30})\n(?P<content>.*?)(?=\n##|\Z)', re.DOTALL),
            # Standard console markdown: **Speaker**: content
            re.compile(r'\*\*(?P<speaker>[A-Za-z0-9_\-\s]{2,30})\*\*\s*:\s*(?P<content>.*?)(?=\n\*\*[A-Za-z0-9_\-\s]{2,30}\*\*|\Z)', re.DOTALL),
            # Magentic-One orchestrator format: Speaker (timestamp): content
            re.compile(r'(?P<speaker>[A-Za-z0-9_\-\s]{2,30})\s*(?:\(\d{2}:\d{2}:\d{2}\))?\s*:\s*(?P<content>.*?)(?=\n[A-Za-z0-9_\-\s]{2,30}\s*(?:\(\d{2}:\d{2}:\d{2}\))?\s*:|\Z)', re.DOTALL),
        ]

    def parse_trace(self, raw_trace: Any, framework_hint: str = "") -> List[Dict[str, Any]]:
        """
        Main entry point: Automatically detects trace format and converts
        it into a normalized list of turns.
        
        Returns:
            List of dicts: [
                {
                    'turn_id': int,
                    'speaker': str,
                    'content': str,
                    'is_goal': bool
                }, ...
            ]
        """
        if not raw_trace:
            return []

        # Strategy 1: If raw_trace is already a Python list or valid dictionary sequence (Common in AG2/AutoGen)
        if isinstance(raw_trace, list):
            return self._parse_dict_list(raw_trace)
        
        # Strategy 2: If it's a string, attempt AST/JSON parsing first (AG2 stored as strings)
        if isinstance(raw_trace, str):
            clean_str = raw_trace.strip()
            
            # Check if string looks like a JSON / Python list: "[{...}, {...}]"
            if clean_str.startswith("[") and clean_str.endswith("]"):
                try:
                    parsed_list = ast.literal_eval(clean_str)
                    if isinstance(parsed_list, list) and len(parsed_list) > 0 and isinstance(parsed_list[0], dict):
                        return self._parse_dict_list(parsed_list)
                except Exception:
                    try:
                        parsed_json = json.loads(clean_str)
                        if isinstance(parsed_json, list):
                            return self._parse_dict_list(parsed_json)
                    except Exception:
                        pass # Fall through to regex text parser

            # Strategy 3: Regex multi-agent terminal/console parsing (ChatDev, MetaGPT, Magentic-One)
            return self._parse_console_text(clean_str)

        return []

    def _parse_dict_list(self, trace_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses structured dictionary lists (AutoGen / AG2 format)."""
        turns = []
        for idx, item in enumerate(trace_list):
            if not isinstance(item, dict):
                continue
            
            # Determine speaker
            speaker = item.get("name") or item.get("role") or item.get("sender") or f"Agent_{idx}"
            
            # Determine content
            content = item.get("content") or item.get("message") or item.get("text") or ""
            if not isinstance(content, str):
                content = str(content)
            
            # First turn containing user prompt is treated as initial task goal s_0
            is_goal = (idx == 0) or (item.get("role") == "user" and idx <= 1)

            turns.append({
                "turn_id": idx,
                "speaker": str(speaker).strip(),
                "content": content.strip(),
                "is_goal": is_goal
            })
        return turns

    def _parse_console_text(self, text: str) -> List[Dict[str, Any]]:
        """Parses unstructured console dumps using regex pattern matching."""
        turns = []
        
        # Try each regex pattern
        for pattern in self.role_header_patterns:
            matches = list(pattern.finditer(text))
            if len(matches) >= 2: # At least 2 turns found
                for idx, match in enumerate(matches):
                    speaker = match.group("speaker").strip()
                    content = match.group("content").strip()
                    
                    # Clean markdown and formatting noise
                    content = re.sub(r'={3,}', '', content).strip()
                    
                    turns.append({
                        "turn_id": idx,
                        "speaker": speaker,
                        "content": content,
                        "is_goal": (idx == 0)
                    })
                return turns

        # Fallback Strategy: If no pattern cleanly matched, chunk by double newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        for idx, para in enumerate(paragraphs):
            # Check if first line contains speaker
            lines = para.split("\n", 1)
            first_line = lines[0].strip()
            
            if ":" in first_line and len(first_line.split(":")[0]) < 25:
                speaker, rest = first_line.split(":", 1)
                content = rest + ("\n" + lines[1] if len(lines) > 1 else "")
            else:
                speaker = f"Agent_{idx}"
                content = para
                
            turns.append({
                "turn_id": idx,
                "speaker": speaker.strip(),
                "content": content.strip(),
                "is_goal": (idx == 0)
            })

        return turns

    def extract_goal_and_messages(self, turns: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts:
        1. Master task specification s_0 (initial goal)
        2. Clean message turns m_t
        """
        if not turns:
            return "", []
        
        # Goal is typically turn 0 or the first marked goal turn
        goal_turn = next((t for t in turns if t["is_goal"]), turns[0])
        s_0 = goal_turn["content"]
        
        # All messages after the goal are the collaborative messages m_t
        m_t = [t for t in turns if t["turn_id"] != goal_turn["turn_id"]]
        
        return s_0, m_t


# =====================================================================
# QUICK SELF-TEST SCRIPT
# =====================================================================
if __name__ == "__main__":
    parser = UniversalTraceParser()

    print("--- Test 1: AutoGen Dictionary Format ---")
    mock_ag2 = [
        {"role": "user", "name": "Admin", "content": "Calculate 25 * 4."},
        {"role": "assistant", "name": "MathAgent", "content": "25 * 4 = 100."},
        {"role": "assistant", "name": "Reviewer", "content": "Verified correct."}
    ]
    turns_ag2 = parser.parse_trace(mock_ag2)
    s0, mt = parser.extract_goal_and_messages(turns_ag2)
    print(f"Goal s_0: {s0}")
    print(f"Parsed {len(mt)} messages successfully!\n")

    print("--- Test 2: ChatDev / MetaGPT Console Log Format ---")
    mock_chatdev = (
        "[User]: Build a snake game in Python.\n"
        "[Programmer]: Here is the code for the game loop.\n"
        "[CodeReviewer]: The game loop has no exit condition."
    )
    turns_chatdev = parser.parse_trace(mock_chatdev)
    s0_c, mt_c = parser.extract_goal_and_messages(turns_chatdev)
    print(f"Goal s_0: {s0_c}")
    print(f"Extracted turns: {[t['speaker'] for t in mt_c]}")
    print("All tests passed with zero crashes!")
