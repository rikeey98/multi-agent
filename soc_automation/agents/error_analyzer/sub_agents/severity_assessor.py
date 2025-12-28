"""
Severity Assessor Sub-Agent.

심각도 재평가 (modifiers 적용)
"""

import json
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from soc_automation.utils.logger import get_agent_logger

logger = get_agent_logger("severity_assessor")


def _load_prompt() -> str:
    """Load severity assessor prompt from MD file."""
    prompt_file = Path(__file__).parent.parent.parent.parent.parent / \
                  "soc_automation/config/prompts/error_analyzer/sub_agents/severity_assessor.md"

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


async def run_severity_assessor(
    llm,
    error_message: str,
    base_severity: int,
    error_type: str,
    component: str = "unknown"
) -> Dict[str, Any]:
    """
    Run severity assessor to calculate final severity with modifiers.

    Args:
        llm: Language model
        error_message: Error message text
        base_severity: Base severity from pattern or 5 for UNKNOWN
        error_type: Pattern code (e.g., "MEM-001") or "UNKNOWN"
        component: Affected component/module name

    Returns:
        {
            "final_severity": int,
            "base_severity": int,
            "modifiers": list[dict],
            "reasoning": str
        }
    """
    logger.info(f"Running severity assessor... base={base_severity}, type={error_type}")

    try:
        # Create agent with severity assessor prompt
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=_load_prompt()
        )

        # Prepare input
        input_msg = f"""
Calculate the final severity for this error:

ERROR MESSAGE:
{error_message}

BASE SEVERITY: {base_severity}
ERROR TYPE: {error_type}
COMPONENT: {component}

Apply modifiers based on:
1. Component impact (critical/high/standard/low priority)
2. Error context (data corruption, system halt, recoverable, etc.)
3. Frequency (if mentioned in error)
4. Timing (early/late in simulation, critical phase)
5. UNKNOWN penalty (if error_type is "UNKNOWN", add +1)

Return a JSON object with:
- final_severity: final severity score (0-10, clamped)
- base_severity: the input base severity
- modifiers: list of applied modifiers with type, value, and reason
- reasoning: detailed explanation of severity calculation

Be conservative. When in doubt, rate higher.
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
                severity_result = json.loads(json_str)
            else:
                # Fallback
                severity_result = {
                    "final_severity": base_severity,
                    "base_severity": base_severity,
                    "modifiers": [],
                    "reasoning": "Failed to parse severity assessment"
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from severity assessor")
            severity_result = {
                "final_severity": base_severity,
                "base_severity": base_severity,
                "modifiers": [],
                "reasoning": f"JSON parse error. Raw: {response_text[:200]}"
            }

        # Validate and clamp
        final_severity = severity_result.get("final_severity", base_severity)
        final_severity = max(0, min(10, final_severity))  # Clamp to [0, 10]
        severity_result["final_severity"] = final_severity

        logger.info(f"Severity assessment: {base_severity} → {final_severity}")

        return severity_result

    except Exception as e:
        logger.error(f"Severity assessor failed: {e}", exc_info=True)
        return {
            "final_severity": base_severity,
            "base_severity": base_severity,
            "modifiers": [],
            "reasoning": f"Error during assessment: {str(e)}"
        }


if __name__ == "__main__":
    """Test severity assessor independently."""
    import os
    import asyncio
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Severity Assessor Sub-Agent Test ===\n")

    # Test cases
    test_cases = [
        {
            "error_message": "Cache coherency violation in CPU core",
            "base_severity": 9,
            "error_type": "MEM-001",
            "component": "cpu_core"
        },
        {
            "error_message": "Unknown strange behavior in peripheral module",
            "base_severity": 5,
            "error_type": "UNKNOWN",
            "component": "peripheral_xyz"
        },
        {
            "error_message": "FIFO overflow with data corruption risk",
            "base_severity": 7,
            "error_type": "DAT-002",
            "component": "tx_fifo"
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
            print("-" * 80)

            result = await run_severity_assessor(
                llm=llm,
                error_message=case['error_message'],
                base_severity=case['base_severity'],
                error_type=case['error_type'],
                component=case['component']
            )

            print(f"Base Severity: {result['base_severity']}")
            print(f"Final Severity: {result['final_severity']}")
            print(f"Modifiers: {len(result['modifiers'])}")
            for mod in result['modifiers']:
                print(f"  - {mod.get('type')}: {mod.get('value'):+d} ({mod.get('reason')})")
            print(f"Reasoning: {result['reasoning']}")
            print()

    asyncio.run(test())
    print("\nTest completed.")
