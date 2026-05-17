"""Load check-worthiness prompts from prompts/checkworthiness_v4_prompts.md.

The markdown file is the canonical source. This module parses it and
exposes the prompts as module-level constants for the inference script.
"""

from pathlib import Path
import re

PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "checkworthiness_v4_prompts.md"


def _parse_prompts(md_text: str) -> dict:
    """Extract the three (system, user, prefill) tuples from the markdown."""
    sections = {}
    current_dim = None
    current_role = None
    in_code_block = False
    buffer = []

    for line in md_text.splitlines():
        dim_match = re.match(r'^## (Checkability|Verifiability|Harm Potential)\s*$', line)
        if dim_match:
            current_dim = dim_match.group(1).lower().replace(" ", "_")
            sections[current_dim] = {}
            continue

        role_match = re.match(r'^### (System|User|Assistant prefill)\s*$', line)
        if role_match:
            current_role = role_match.group(1).lower().replace(" prefill", "_prefill").replace(" ", "_")
            continue

        if line.startswith("```"):
            if in_code_block:
                if current_dim and current_role:
                    sections[current_dim][current_role] = "\n".join(buffer)
                buffer = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            buffer.append(line)

    return sections


_prompts = _parse_prompts(PROMPTS_PATH.read_text(encoding="utf-8"))

CHECKABILITY_SYSTEM   = _prompts["checkability"]["system"]
CHECKABILITY_USER     = _prompts["checkability"]["user"].replace("{", "{{").replace("}", "}}").replace("{{claim}}", "{claim}")
CHECKABILITY_PREFILL  = _prompts["checkability"]["assistant_prefill"]

VERIFIABILITY_SYSTEM  = _prompts["verifiability"]["system"]
VERIFIABILITY_USER    = _prompts["verifiability"]["user"].replace("{", "{{").replace("}", "}}").replace("{{claim}}", "{claim}")
VERIFIABILITY_PREFILL = _prompts["verifiability"]["assistant_prefill"]

HARM_SYSTEM           = _prompts["harm_potential"]["system"]
HARM_USER             = _prompts["harm_potential"]["user"].replace("{", "{{").replace("}", "}}").replace("{{claim}}", "{claim}")
HARM_PREFILL          = _prompts["harm_potential"]["assistant_prefill"]

assert all([CHECKABILITY_SYSTEM, CHECKABILITY_USER, CHECKABILITY_PREFILL,
            VERIFIABILITY_SYSTEM, VERIFIABILITY_USER, VERIFIABILITY_PREFILL,
            HARM_SYSTEM, HARM_USER, HARM_PREFILL]), \
    "Prompt parsing failed — check prompts/checkworthiness_v4_prompts.md format"
