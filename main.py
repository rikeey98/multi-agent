"""
Main entry point for SOC Automation Multi-Agent System.

SOC 검증 자동화 Multi-Agent 시스템의 메인 진입점
- LangGraph Supervisor 패턴 사용
- 7개 Agent 통합 실행
- MCP 서버 통합
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from soc_automation.config.settings import settings
from soc_automation.utils.state import (
    AgentState,
    create_initial_state,
    update_state_with_error_analysis,
    update_state_with_sop_results,
    update_state_with_collected_data,
    update_state_with_resolution_plan,
    update_state_with_execution_result,
    update_state_with_notification,
    ErrorAnalysis,
    ErrorCategory,
)
from soc_automation.utils.logger import get_logger, get_workflow_logger
from soc_automation.agents.error_analyzer import create_error_analyzer_agent, DEFAULT_TOOLS as ERROR_TOOLS
from soc_automation.agents.sop_searcher import create_sop_searcher_agent, DEFAULT_TOOLS as SOP_TOOLS
from soc_automation.agents.data_collector import create_data_collector_agent, DEFAULT_TOOLS as DATA_TOOLS
from soc_automation.agents.decision_maker import create_decision_maker_agent, DEFAULT_TOOLS as DECISION_TOOLS
from soc_automation.agents.auto_executor import create_auto_executor_agent, DEFAULT_TOOLS as EXECUTOR_TOOLS
from soc_automation.agents.notification import create_notification_agent, DEFAULT_TOOLS as NOTIFICATION_TOOLS
from soc_automation.mcp_client import MCPClientLoader
from soc_automation.mcp_client.helpers import filter_enabled_servers


logger = get_logger()

# Global MCP tools storage
_mcp_tools = []


async def error_analyzer_node(state: AgentState) -> AgentState:
    """
    Error analyzer node.

    에러 분석 노드
    """
    logger.info("Running error analyzer...")

    try:
        # Create LLM
        llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=settings.openai.temperature,
            api_key=settings.openai.api_key
        )

        # Combine default tools with MCP tools
        all_tools = list(ERROR_TOOLS) + _mcp_tools

        # Create agent
        agent = create_error_analyzer_agent(llm, all_tools)

        # Prepare input
        from langchain_core.messages import HumanMessage
        input_msg = f"Analyze log file: {state['log_file_path']}"

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        # Extract analysis (simplified for now - in production, parse the actual output)
        messages = result.get("messages", [])
        analysis_text = messages[-1].content if messages else "No analysis"

        # Create error analysis (simplified)
        error_analysis = ErrorAnalysis(
            error_type=ErrorCategory.UNKNOWN,  # Should be parsed from agent output
            severity=5,  # Should be parsed from agent output
            error_message=analysis_text[:200],
            location="unknown",
            timestamp=datetime.now().isoformat(),
            context="",
            pattern_match=None,
            root_cause_hypothesis="Pending analysis",
            affected_modules=[]
        )

        # Update state
        state = update_state_with_error_analysis(state, error_analysis)
        logger.info("Error analysis completed")

        return state

    except Exception as e:
        logger.error(f"Error analyzer failed: {e}", exc_info=True)
        state["errors"].append(f"Error analyzer failed: {str(e)}")
        state["workflow_status"] = "FAILED"
        return state


async def sop_searcher_node(state: AgentState) -> AgentState:
    """
    SOP searcher node.

    SOP 검색 노드
    """
    logger.info("Running SOP searcher...")

    try:
        # Create LLM
        llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=settings.openai.temperature,
            api_key=settings.openai.api_key
        )

        # Combine default tools with MCP tools
        all_tools = list(SOP_TOOLS) + _mcp_tools

        # Create agent
        agent = create_sop_searcher_agent(llm, all_tools)

        # Prepare input
        from langchain_core.messages import HumanMessage
        error_type = state.get("error_analysis", {}).get("error_type", "UNKNOWN")
        error_msg = state.get("error_analysis", {}).get("error_message", "")

        input_msg = f"""
Search SOP for:
Error Type: {error_type}
Error Message: {error_msg}
SOP Directory: {settings.paths.sop_dir}
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        logger.info("SOP search completed")
        return state

    except Exception as e:
        logger.error(f"SOP searcher failed: {e}", exc_info=True)
        state["errors"].append(f"SOP searcher failed: {str(e)}")
        return state


async def data_collector_node(state: AgentState) -> AgentState:
    """
    Data collector node.

    데이터 수집 노드
    """
    logger.info("Running data collector...")

    try:
        # Create LLM
        llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=settings.openai.temperature,
            api_key=settings.openai.api_key
        )

        # Combine default tools with MCP tools
        all_tools = list(DATA_TOOLS) + _mcp_tools

        # Create agent
        agent = create_data_collector_agent(llm, all_tools)

        # Prepare input
        from langchain_core.messages import HumanMessage
        error_type = state.get("error_analysis", {}).get("error_type", "UNKNOWN")

        input_msg = f"""
Collect data for:
Error Type: {error_type}
Log File: {state['log_file_path']}
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        logger.info("Data collection completed")
        return state

    except Exception as e:
        logger.error(f"Data collector failed: {e}", exc_info=True)
        state["errors"].append(f"Data collector failed: {str(e)}")
        return state


async def decision_maker_node(state: AgentState) -> AgentState:
    """
    Decision maker node.

    의사결정 노드
    """
    logger.info("Running decision maker...")

    try:
        # Create LLM
        llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=settings.openai.temperature,
            api_key=settings.openai.api_key
        )

        # Combine default tools with MCP tools
        all_tools = list(DECISION_TOOLS) + _mcp_tools

        # Create agent
        agent = create_decision_maker_agent(llm, all_tools)

        # Prepare input
        from langchain_core.messages import HumanMessage
        import json

        input_msg = f"""
Make decision based on:

Error Analysis: {json.dumps(state.get('error_analysis', {}), default=str)}
SOP Results: {json.dumps(state.get('sop_results', []), default=str)}
Collected Data: {json.dumps(state.get('collected_data', {}), default=str)}
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        # Create simplified resolution plan
        from soc_automation.utils.state import ResolutionPlan, RiskLevel
        resolution_plan = ResolutionPlan(
            root_cause="To be determined",
            confidence=0.7,
            resolution_steps=["Step 1", "Step 2"],
            auto_executable=False,  # Conservative default
            risk_level=RiskLevel.MEDIUM,
            required_approvals=["manual_review"],
            rollback_plan=["Rollback step 1"],
            expected_outcome="Error resolved",
            estimated_duration="30 minutes"
        )

        state = update_state_with_resolution_plan(state, resolution_plan)
        logger.info("Decision making completed")

        return state

    except Exception as e:
        logger.error(f"Decision maker failed: {e}", exc_info=True)
        state["errors"].append(f"Decision maker failed: {str(e)}")
        return state


async def auto_executor_node(state: AgentState) -> AgentState:
    """
    Auto executor node.

    자동 실행 노드
    """
    logger.info("Running auto executor...")

    try:
        # Check if auto-executable
        resolution_plan = state.get("resolution_plan", {})
        if not resolution_plan.get("auto_executable", False):
            logger.info("Skipping auto execution - not approved")
            state["next_agent"] = "notification"
            return state

        # Create LLM
        llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=settings.openai.temperature,
            api_key=settings.openai.api_key
        )

        # Combine default tools with MCP tools
        all_tools = list(EXECUTOR_TOOLS) + _mcp_tools

        # Create agent
        agent = create_auto_executor_agent(llm, all_tools)

        # Prepare input
        from langchain_core.messages import HumanMessage
        import json

        input_msg = f"""
Execute resolution plan:
{json.dumps(resolution_plan, default=str)}
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        logger.info("Auto execution completed")
        state["next_agent"] = "notification"
        return state

    except Exception as e:
        logger.error(f"Auto executor failed: {e}", exc_info=True)
        state["errors"].append(f"Auto executor failed: {str(e)}")
        state["next_agent"] = "notification"
        return state


async def notification_node(state: AgentState) -> AgentState:
    """
    Notification node.

    알림 노드
    """
    logger.info("Running notification agent...")

    try:
        # Create LLM
        llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=settings.openai.temperature,
            api_key=settings.openai.api_key
        )

        # Combine default tools with MCP tools
        all_tools = list(NOTIFICATION_TOOLS) + _mcp_tools

        # Create agent
        agent = create_notification_agent(llm, all_tools)

        # Prepare input
        from langchain_core.messages import HumanMessage
        import json

        input_msg = f"""
Generate notification for:

Workflow ID: {state['workflow_id']}
Error Analysis: {json.dumps(state.get('error_analysis', {}), default=str)}
Resolution Plan: {json.dumps(state.get('resolution_plan', {}), default=str)}
Execution Result: {json.dumps(state.get('execution_result', {}), default=str)}
"""

        # Run agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input_msg)]})

        logger.info("Notification sent")

        # Mark workflow as completed
        state["workflow_status"] = "COMPLETED"
        state["completed_at"] = datetime.now().isoformat()
        state["next_agent"] = "END"

        return state

    except Exception as e:
        logger.error(f"Notification failed: {e}", exc_info=True)
        state["errors"].append(f"Notification failed: {str(e)}")
        state["workflow_status"] = "FAILED"
        state["next_agent"] = "END"
        return state


def should_continue(state: AgentState) -> str:
    """
    Determine next node based on state.

    상태에 따라 다음 노드를 결정합니다.
    """
    next_agent = state.get("next_agent", "END")

    if state.get("workflow_status") == "FAILED":
        return "notification"

    if next_agent == "parallel_collection":
        return "parallel_collection"
    elif next_agent == "decision_maker":
        return "decision_maker"
    elif next_agent == "auto_executor":
        return "auto_executor"
    elif next_agent == "notification":
        return "notification"
    else:
        return END


def create_workflow() -> StateGraph:
    """
    Create LangGraph workflow.

    LangGraph 워크플로우를 생성합니다.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("error_analyzer", error_analyzer_node)
    workflow.add_node("sop_searcher", sop_searcher_node)
    workflow.add_node("data_collector", data_collector_node)
    workflow.add_node("decision_maker", decision_maker_node)
    workflow.add_node("auto_executor", auto_executor_node)
    workflow.add_node("notification", notification_node)

    # Set entry point
    workflow.set_entry_point("error_analyzer")

    # Add edges
    workflow.add_edge("error_analyzer", "sop_searcher")
    workflow.add_edge("sop_searcher", "data_collector")
    workflow.add_edge("data_collector", "decision_maker")

    # Conditional edges from decision_maker
    workflow.add_conditional_edges(
        "decision_maker",
        lambda state: "auto_executor" if state.get("resolution_plan", {}).get("auto_executable") else "notification"
    )

    workflow.add_edge("auto_executor", "notification")
    workflow.add_edge("notification", END)

    return workflow


async def run_workflow(log_file_path: str, trigger_event: str = "manual", use_mcp: bool = True) -> dict:
    """
    Run the complete workflow.

    전체 워크플로우를 실행합니다.

    Args:
        log_file_path: Path to log file
        trigger_event: Event that triggered the workflow
        use_mcp: Whether to load MCP tools (default: True)

    Returns:
        dict: Workflow result
    """
    global _mcp_tools

    logger.info(f"Starting workflow for: {log_file_path}")

    # Create initial state
    initial_state = create_initial_state(log_file_path, trigger_event)

    # Create workflow logger
    workflow_logger = get_workflow_logger(initial_state["workflow_id"])
    workflow_logger.info(f"Workflow started for: {log_file_path}")

    # Load MCP tools if enabled
    mcp_loader = None
    if use_mcp:
        try:
            logger.info("Loading MCP tools...")
            mcp_loader = MCPClientLoader()
            await mcp_loader.load_all_servers()
            _mcp_tools = mcp_loader.get_all_tools()
            logger.info(f"Loaded {len(_mcp_tools)} MCP tools")
        except Exception as e:
            logger.warning(f"Failed to load MCP tools: {e}")
            logger.warning("Continuing without MCP tools")
            _mcp_tools = []

    try:
        # Create workflow
        workflow = create_workflow()

        # Compile with checkpointer
        app = workflow.compile(checkpointer=MemorySaver())

        # Run workflow
        config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
        final_state = await app.ainvoke(initial_state, config)

        workflow_logger.info(f"Workflow completed with status: {final_state['workflow_status']}")

        return {
            "status": "success",
            "workflow_id": final_state["workflow_id"],
            "workflow_status": final_state["workflow_status"],
            "errors": final_state.get("errors", []),
            "final_state": final_state
        }

    except Exception as e:
        workflow_logger.error(f"Workflow failed: {e}", exc_info=True)
        return {
            "status": "error",
            "workflow_id": initial_state["workflow_id"],
            "error": str(e)
        }
    finally:
        # Clean up MCP connections
        if mcp_loader:
            try:
                await mcp_loader.close()
                logger.info("MCP connections closed")
            except Exception as e:
                logger.warning(f"Error closing MCP connections: {e}")


async def main_async(args):
    """
    Main async function.

    메인 비동기 함수
    """
    log_file = args.log_file

    if not Path(log_file).exists():
        logger.error(f"Log file not found: {log_file}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("SOC Automation Multi-Agent System")
    logger.info("=" * 80)
    logger.info(f"Log File: {log_file}")
    logger.info(f"Model: {settings.openai.model}")
    logger.info(f"MCP Tools: {'Enabled' if not args.no_mcp else 'Disabled'}")
    logger.info("=" * 80)

    # Run workflow
    result = await run_workflow(log_file, trigger_event="cli", use_mcp=not args.no_mcp)

    # Print result
    logger.info("\n" + "=" * 80)
    logger.info("Workflow Result")
    logger.info("=" * 80)
    logger.info(f"Status: {result['status']}")
    logger.info(f"Workflow ID: {result['workflow_id']}")

    if result['status'] == 'success':
        logger.info(f"Workflow Status: {result['workflow_status']}")
        if result.get('errors'):
            logger.warning(f"Errors encountered: {len(result['errors'])}")
            for error in result['errors']:
                logger.warning(f"  - {error}")
    else:
        logger.error(f"Error: {result.get('error')}")

    logger.info("=" * 80)


def main():
    """
    Main entry point.

    메인 진입점
    """
    parser = argparse.ArgumentParser(
        description="SOC Automation Multi-Agent System"
    )
    parser.add_argument(
        "log_file",
        help="Path to log file to analyze"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model to use (overrides env)"
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP tools loading"
    )

    args = parser.parse_args()

    # Override model if specified
    if args.model:
        settings.openai.model = args.model

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
