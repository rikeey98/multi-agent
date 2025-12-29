"""
Pattern Matcher Sub-Agent.

100+ 회사 패턴 매칭
"""

import json
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from soc_automation.utils.logger import get_agent_logger

logger = get_agent_logger("pattern_matcher")


def _load_prompt() -> str:
    """Load pattern matcher prompt from MD file."""
    prompt_file = Path(__file__).parent.parent.parent.parent.parent / \
                  "soc_automation/config/prompts/error_analyzer/sub_agents/pattern_matcher.md"

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


async def run_pattern_matcher(
    llm,
    error_message: str,
    log_context: str = ""
) -> Dict[str, Any]:
    """
    Run pattern matcher to match error against 100+ company patterns.

    Args:
        llm: Language model
        error_message: Error message to match
        log_context: Additional log context (optional)

    Returns:
        {
            "matched": bool,
            "pattern_code": str | None,
            "pattern_name": str | None,
            "base_severity": int | None,
            "confidence": float,
            "reasoning": str
        }
    """
    logger.info("Running pattern matcher...")

    try:
        # Create agent with pattern matcher prompt
        agent = create_agent(
            model=llm,
            tools=[],  # No tools needed for pattern matching
            system_prompt=_load_prompt()
        )

        # Prepare input
        input_msg = f"""
Match this error message against company error patterns:

ERROR MESSAGE:
{error_message}

ADDITIONAL CONTEXT:
{log_context if log_context else "No additional context"}

Analyze the error and return a JSON object with:
- matched: true if pattern found with confidence > 0.7, false otherwise
- pattern_code: the pattern code (e.g., "MEM-001") or null
- pattern_name: the pattern name or null
- base_severity: the base severity (0-10) or null
- confidence: confidence score (0.0-1.0)
- reasoning: explanation of the match or why no match

Be precise. Only match if you are confident (> 0.7).
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        # Extract result from agent output
        messages = result.get("messages", [])
        response_text = messages[-1].content if messages else "{}"

        # Try to parse JSON from response
        try:
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                pattern_result = json.loads(json_str)
            else:
                # Fallback: no match
                pattern_result = {
                    "matched": False,
                    "pattern_code": None,
                    "pattern_name": None,
                    "base_severity": None,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse pattern match result"
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from pattern matcher response")
            pattern_result = {
                "matched": False,
                "pattern_code": None,
                "pattern_name": None,
                "base_severity": None,
                "confidence": 0.0,
                "reasoning": f"JSON parse error. Raw response: {response_text[:200]}"
            }

        logger.info(f"Pattern match result: matched={pattern_result.get('matched')}, "
                   f"pattern={pattern_result.get('pattern_code')}")

        return pattern_result

    except Exception as e:
        logger.error(f"Pattern matcher failed: {e}", exc_info=True)
        return {
            "matched": False,
            "pattern_code": None,
            "pattern_name": None,
            "base_severity": None,
            "confidence": 0.0,
            "reasoning": f"Error during pattern matching: {str(e)}"
        }


if __name__ == "__main__":
    """Test pattern matcher independently."""
    import os
    import asyncio
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Pattern Matcher Sub-Agent Test ===\n")

    # Test cases
    test_cases = [
        "Cache coherency violation detected at address 0x1000",
        "AXI SLVERR on write transaction to 0x2000",
        "Simulation timeout after 10000000 cycles",
        "Unknown strange error in module XYZ",
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
        for i, error_msg in enumerate(test_cases, 1):
            print(f"\nTest {i}: {error_msg}")
            print("-" * 80)

            result = await run_pattern_matcher(llm, error_msg)

            print(f"Matched: {result['matched']}")
            print(f"Pattern: {result['pattern_code']} - {result['pattern_name']}")
            print(f"Base Severity: {result['base_severity']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Reasoning: {result['reasoning']}")
            print()

    asyncio.run(test())
    print("\nTest completed.")
