"""
Decision Maker Agent for SOC automation system.

의사결정 Agent
- 모든 정보를 종합하여 해결 계획 수립
- 자동 실행 가능 여부 판단
- 위험도 평가
"""

import asyncio
from typing import List, Dict, Any, Optional
import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..config.agent_prompts import get_agent_prompt
from ..utils.state import ResolutionPlan, RiskLevel
from ..utils.logger import get_agent_logger


# Tools for Decision Maker Agent
@tool
def analyze_risk_level(operation_description: str) -> str:
    """
    Analyze risk level of proposed operations.

    제안된 작업의 위험 수준을 분석합니다.

    Args:
        operation_description: Description of operations to perform

    Returns:
        str: Risk analysis result
    """
    try:
        desc_lower = operation_description.lower()

        # High risk operations
        high_risk_keywords = [
            "delete database", "drop table", "truncate", "rm -rf",
            "modify schema", "alter table", "delete *", "format",
            "network change", "firewall", "reboot", "shutdown"
        ]

        # Medium risk operations
        medium_risk_keywords = [
            "delete file", "modify config", "restart service",
            "kill process", "change parameter", "update setting"
        ]

        # Low risk operations
        low_risk_keywords = [
            "read", "check", "verify", "analyze", "log",
            "clear cache", "cleanup temp", "increase timeout"
        ]

        high_risk_count = sum(1 for kw in high_risk_keywords if kw in desc_lower)
        medium_risk_count = sum(1 for kw in medium_risk_keywords if kw in desc_lower)
        low_risk_count = sum(1 for kw in low_risk_keywords if kw in desc_lower)

        if high_risk_count > 0:
            risk_level = "HIGH"
            reason = f"Contains high-risk operations (found {high_risk_count} high-risk keywords)"
            auto_executable = False
        elif medium_risk_count > 1:
            risk_level = "MEDIUM"
            reason = f"Contains multiple medium-risk operations (found {medium_risk_count} keywords)"
            auto_executable = False
        elif medium_risk_count == 1:
            risk_level = "MEDIUM"
            reason = "Contains single medium-risk operation - requires review"
            auto_executable = True  # Can auto-execute with proper backup
        else:
            risk_level = "LOW"
            reason = "Low-risk operations only"
            auto_executable = True

        return f"""
Risk Analysis:
- Risk Level: {risk_level}
- Auto Executable: {auto_executable}
- Reason: {reason}
- High-risk operations: {high_risk_count}
- Medium-risk operations: {medium_risk_count}
- Low-risk operations: {low_risk_count}
"""

    except Exception as e:
        return f"Error analyzing risk: {str(e)}"


@tool
def evaluate_solution_confidence(
    error_pattern_match: str,
    sop_relevance: str,
    historical_success_rate: Optional[str] = None
) -> str:
    """
    Evaluate confidence in the proposed solution.

    제안된 솔루션에 대한 신뢰도를 평가합니다.

    Args:
        error_pattern_match: Whether error pattern was matched (yes/no)
        sop_relevance: SOP relevance score (0-1)
        historical_success_rate: Historical success rate if available

    Returns:
        str: Confidence evaluation
    """
    try:
        confidence_score = 0.0

        # Pattern match contributes 40%
        if error_pattern_match.lower() == "yes":
            confidence_score += 0.4

        # SOP relevance contributes 40%
        try:
            relevance = float(sop_relevance)
            confidence_score += 0.4 * relevance
        except Exception:
            pass

        # Historical success contributes 20%
        if historical_success_rate:
            try:
                success_rate = float(historical_success_rate)
                confidence_score += 0.2 * success_rate
            except Exception:
                pass

        if confidence_score >= 0.8:
            confidence_level = "HIGH"
        elif confidence_score >= 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return f"""
Solution Confidence Evaluation:
- Confidence Score: {confidence_score:.2f}
- Confidence Level: {confidence_level}
- Pattern Match: {error_pattern_match}
- SOP Relevance: {sop_relevance}
- Historical Success: {historical_success_rate or 'N/A'}

Recommendation: {'Proceed with automation' if confidence_score >= 0.6 else 'Manual review recommended'}
"""

    except Exception as e:
        return f"Error evaluating confidence: {str(e)}"


@tool
def generate_rollback_plan(proposed_actions: str) -> str:
    """
    Generate rollback plan for proposed actions.

    제안된 작업에 대한 롤백 계획을 생성합니다.

    Args:
        proposed_actions: Description of proposed actions

    Returns:
        str: Rollback plan
    """
    try:
        actions_lower = proposed_actions.lower()
        rollback_steps = []

        # Common rollback strategies
        if "modify" in actions_lower or "update" in actions_lower or "change" in actions_lower:
            rollback_steps.append("1. Restore configuration from backup")
            rollback_steps.append("2. Verify original settings restored")

        if "delete" in actions_lower:
            rollback_steps.append("1. Restore deleted files from backup")
            rollback_steps.append("2. Verify file integrity")

        if "restart" in actions_lower:
            rollback_steps.append("1. No rollback needed - service restart is reversible")

        if "increase" in actions_lower or "decrease" in actions_lower:
            rollback_steps.append("1. Revert parameter to original value")
            rollback_steps.append("2. Restart affected services if needed")

        if not rollback_steps:
            rollback_steps.append("1. Manual review required for rollback")
            rollback_steps.append("2. Consult system administrator")

        # Add common final steps
        rollback_steps.append("3. Verify system state after rollback")
        rollback_steps.append("4. Review logs for any issues")

        return "Rollback Plan:\n" + '\n'.join(rollback_steps)

    except Exception as e:
        return f"Error generating rollback plan: {str(e)}"


# Default tools for decision maker
DEFAULT_TOOLS = [
    analyze_risk_level,
    evaluate_solution_confidence,
    generate_rollback_plan,
]


def create_decision_maker_agent(llm, tools: Optional[List] = None):
    """
    Create Decision Maker Agent.

    의사결정 Agent를 생성합니다.

    Args:
        llm: Language model to use
        tools: List of tools (default: DEFAULT_TOOLS)

    Returns:
        Agent: Decision Maker Agent
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=get_agent_prompt("decision_maker")
    )

    return agent


async def run_decision_maker(
    llm,
    error_analysis: Dict[str, Any],
    sop_results: Dict[str, Any],
    collected_data: Dict[str, Any],
    tools: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run decision maker agent.

    의사결정 Agent를 실행합니다.

    Args:
        llm: Language model
        error_analysis: Error analysis results
        sop_results: SOP search results
        collected_data: Collected data
        tools: Optional tools list

    Returns:
        Dict: Resolution plan
    """
    logger = get_agent_logger("decision_maker")
    logger.info("Starting decision making process")

    try:
        # Create agent
        agent = create_decision_maker_agent(llm, tools)

        # Prepare input message
        input_message = f"""
Please analyze all available information and make a decision on the resolution plan:

=== Error Analysis ===
{json.dumps(error_analysis, indent=2)}

=== SOP Results ===
{json.dumps(sop_results, indent=2)}

=== Collected Data ===
{json.dumps(collected_data, indent=2)}

Tasks:
1. Correlate error with system state and historical patterns
2. Evaluate available SOP procedures and their applicability
3. Assess automation feasibility using risk analysis
4. Determine confidence level in the solution
5. Generate rollback plan for proposed actions
6. Make final decision on auto-execution

Provide a structured resolution plan with:
- root_cause: Identified root cause
- confidence: Confidence level (0-1)
- resolution_steps: Step-by-step action plan
- auto_executable: Boolean flag (true/false)
- risk_level: LOW/MEDIUM/HIGH
- required_approvals: List of required approvals (empty if auto-executable)
- rollback_plan: Rollback procedure
- expected_outcome: Expected results after execution
- estimated_duration: Estimated time to resolve

Make conservative decisions. When in doubt, require manual review.
"""

        # Run agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]}
        )

        logger.info("Decision making completed")

        # Extract results from agent output
        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else "No decision made"

        return {
            "status": "success",
            "resolution_plan": final_message,
            "raw_result": result
        }

    except Exception as e:
        logger.error(f"Decision making failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    """
    Test Decision Maker Agent independently.

    독립적으로 Decision Maker Agent를 테스트합니다.
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Decision Maker Agent Test ===\n")

    # Sample input data
    error_analysis = {
        "error_type": "TIMEOUT",
        "severity": 7,
        "error_message": "Simulation timeout after 3600 seconds",
        "pattern_match": "TIMEOUT_001"
    }

    sop_results = {
        "sop_id": "SOP-001",
        "title": "Timeout Resolution",
        "relevance_score": 0.95,
        "resolution_steps": [
            "Check for infinite loops",
            "Increase timeout in config",
            "Rerun simulation"
        ],
        "automation_feasible": True
    }

    collected_data = {
        "system_status": "Normal - CPU: 30%, Memory: 50%",
        "running_processes": "1 simulation process found",
        "recent_changes": "Config file modified 2 hours ago"
    }

    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        print("Please set OPENAI_API_KEY in .env file")
        exit(1)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7
    )

    # Run decision maker
    async def test():
        result = await run_decision_maker(
            llm=llm,
            error_analysis=error_analysis,
            sop_results=sop_results,
            collected_data=collected_data
        )

        print("\n=== Decision Result ===\n")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"\nResolution Plan:\n{result['resolution_plan']}")
        else:
            print(f"Error: {result.get('error')}")

    # Run test
    asyncio.run(test())

    print("\nTest completed.")
