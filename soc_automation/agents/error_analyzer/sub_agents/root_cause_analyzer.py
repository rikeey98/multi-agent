"""
Root Cause Analyzer Sub-Agent.

근본 원인 분석 및 권장 조치
"""

import json
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from soc_automation.utils.logger import get_agent_logger

logger = get_agent_logger("root_cause_analyzer")


def _load_prompt() -> str:
    """Load root cause analyzer prompt from MD file."""
    prompt_file = Path(__file__).parent.parent.parent.parent.parent / \
                  "soc_automation/config/prompts/error_analyzer/sub_agents/root_cause_analyzer.md"

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


async def run_root_cause_analyzer(
    llm,
    error_message: str,
    error_type: str,
    severity: int
) -> Dict[str, Any]:
    """
    Run root cause analyzer to hypothesize root cause and recommend actions.

    Args:
        llm: Language model
        error_message: Error message text
        error_type: Pattern code (e.g., "MEM-001") or "UNKNOWN"
        severity: Final severity score

    Returns:
        {
            "hypothesis": str,
            "confidence": float,
            "evidence": list[str],
            "alternative_causes": list[str],
            "recommended_actions": list[str],
            "similar_cases": list[str],
            "needs_new_pattern": bool
        }
    """
    logger.info(f"Running root cause analyzer... type={error_type}, severity={severity}")

    try:
        # Create agent with root cause analyzer prompt
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=_load_prompt()
        )

        # Special handling for UNKNOWN errors
        unknown_note = ""
        if error_type == "UNKNOWN":
            unknown_note = """
**IMPORTANT**: This is an UNKNOWN error.
- Search by KEYWORDS, not pattern code
- Set needs_new_pattern to true
- Suggest pattern characteristics (category, keywords, base severity)
- Recommend adding to pattern database
"""

        # Prepare input
        input_msg = f"""
Analyze the root cause of this error:

ERROR MESSAGE:
{error_message}

ERROR TYPE: {error_type}
SEVERITY: {severity}

{unknown_note}

Return a JSON object with:
- hypothesis: primary root cause hypothesis (string)
- confidence: confidence in hypothesis (0.0-1.0)
- evidence: list of supporting evidence (list of strings)
- alternative_causes: list of alternative explanations (list of strings)
- recommended_actions: list of recommended actions (list of strings)
- similar_cases: list of similar historical cases (list of strings)
- needs_new_pattern: true if UNKNOWN error, false otherwise (boolean)

If error_type is "UNKNOWN", also include:
- suggested_pattern: object with category, pattern_code, keywords, base_severity, pattern_name

Provide actionable recommendations and clear reasoning.
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        # Extract result
        messages = result.get("messages", [])
        response_text = messages[-1].content if messages else "{}"

        # Parse JSON
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                root_cause_result = json.loads(json_str)
            else:
                # Fallback
                root_cause_result = {
                    "hypothesis": "Unable to determine root cause",
                    "confidence": 0.0,
                    "evidence": [],
                    "alternative_causes": [],
                    "recommended_actions": ["Manual investigation required"],
                    "similar_cases": [],
                    "needs_new_pattern": error_type == "UNKNOWN"
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from root cause analyzer")
            root_cause_result = {
                "hypothesis": f"Parse error. Raw: {response_text[:200]}",
                "confidence": 0.0,
                "evidence": [],
                "alternative_causes": [],
                "recommended_actions": ["Check raw output"],
                "similar_cases": [],
                "needs_new_pattern": error_type == "UNKNOWN"
            }

        # Ensure needs_new_pattern is set correctly for UNKNOWN
        if error_type == "UNKNOWN":
            root_cause_result["needs_new_pattern"] = True

        logger.info(f"Root cause analysis: confidence={root_cause_result.get('confidence')}, "
                   f"needs_pattern={root_cause_result.get('needs_new_pattern')}")

        return root_cause_result

    except Exception as e:
        logger.error(f"Root cause analyzer failed: {e}", exc_info=True)
        return {
            "hypothesis": f"Error during analysis: {str(e)}",
            "confidence": 0.0,
            "evidence": [],
            "alternative_causes": [],
            "recommended_actions": ["Fix analyzer error"],
            "similar_cases": [],
            "needs_new_pattern": error_type == "UNKNOWN"
        }


if __name__ == "__main__":
    """Test root cause analyzer independently."""
    import os
    import asyncio
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Root Cause Analyzer Sub-Agent Test ===\n")

    # Test cases
    test_cases = [
        {
            "error_message": "Cache coherency violation in L2 cache controller",
            "error_type": "MEM-001",
            "severity": 9
        },
        {
            "error_message": "Unexpected stall in pipeline stage 3",
            "error_type": "UNKNOWN",
            "severity": 7
        },
        {
            "error_message": "AXI SLVERR on write to 0x2000",
            "error_type": "BUS-001",
            "severity": 8
        },
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        exit(1)

    llm_kwargs = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.7
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**llm_kwargs)

    async def test():
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {case['error_message']}")
            print(f"Type: {case['error_type']}, Severity: {case['severity']}")
            print("-" * 80)

            result = await run_root_cause_analyzer(
                llm=llm,
                error_message=case['error_message'],
                error_type=case['error_type'],
                severity=case['severity']
            )

            print(f"Hypothesis: {result['hypothesis']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Needs New Pattern: {result['needs_new_pattern']}")
            print(f"\nEvidence:")
            for ev in result.get('evidence', []):
                print(f"  - {ev}")
            print(f"\nRecommended Actions:")
            for action in result.get('recommended_actions', []):
                print(f"  - {action}")

            if result.get('suggested_pattern'):
                print(f"\nSuggested Pattern:")
                sp = result['suggested_pattern']
                print(f"  Category: {sp.get('category')}")
                print(f"  Code: {sp.get('pattern_code')}")
                print(f"  Keywords: {sp.get('keywords')}")
            print()

    asyncio.run(test())
    print("\nTest completed.")
