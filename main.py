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
from soc_automation.utils.workflow_storage import get_workflow_storage
from soc_automation.agents.error_analyzer.main import run_error_analyzer
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
    Error analyzer node using sub-agents.

    에러 분석 노드 (3개 Sub-Agent 사용)
    - Pattern Matcher: 100+ 패턴 매칭
    - Severity Assessor: 심각도 재평가
    - Root Cause Analyzer: 근본 원인 분석
    """
    logger.info("Running error analyzer with sub-agents...")

    try:
        # Create LLM
        llm_kwargs = {
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "api_key": settings.openai.api_key
        }
        if settings.openai.base_url:
            llm_kwargs["base_url"] = settings.openai.base_url
        llm = ChatOpenAI(**llm_kwargs)

        # Run new sub-agent based error analyzer
        result = await run_error_analyzer(
            llm=llm,
            log_file_path=state['log_file_path'],
            tools=_mcp_tools
        )

        # Check if analysis succeeded
        if result["status"] == "error":
            logger.error(f"Error analysis failed: {result.get('error')}")
            state["errors"].append(f"Error analyzer failed: {result.get('error')}")
            state["workflow_status"] = "FAILED"
            return state

        # Map error_type to ErrorCategory
        error_type_str = result.get("error_type", "UNKNOWN")
        try:
            # Try to map to ErrorCategory enum
            if error_type_str.startswith("MEM"):
                error_category = ErrorCategory.MEMORY
            elif error_type_str.startswith("BUS"):
                error_category = ErrorCategory.BUS_PROTOCOL
            elif error_type_str.startswith("TIM"):
                error_category = ErrorCategory.TIMEOUT
            elif error_type_str.startswith("AST"):
                error_category = ErrorCategory.ASSERTION
            elif error_type_str.startswith("CFG"):
                error_category = ErrorCategory.CONFIGURATION
            elif error_type_str.startswith("PRT"):
                error_category = ErrorCategory.PROTOCOL
            elif error_type_str.startswith("CLK") or error_type_str.startswith("PWR"):
                error_category = ErrorCategory.CLOCK_DOMAIN
            else:
                error_category = ErrorCategory.UNKNOWN
        except:
            error_category = ErrorCategory.UNKNOWN

        # Create ErrorAnalysis object from sub-agent results
        error_analysis = ErrorAnalysis(
            error_type=error_category,
            severity=result.get("severity", 5),
            error_message=result.get("error_message", ""),
            location=result.get("component", "unknown"),
            timestamp=datetime.now().isoformat(),
            context=result.get("pattern_result", {}).get("reasoning", ""),
            pattern_match=result.get("pattern_result", {}).get("pattern_code") if result.get("pattern_matched") else None,
            root_cause_hypothesis=result.get("root_cause_result", {}).get("hypothesis", ""),
            affected_modules=[result.get("component")] if result.get("component") else []
        )

        # Update state
        state = update_state_with_error_analysis(state, error_analysis)

        # Log completion with details
        logger.info(f"Error analysis completed:")
        logger.info(f"  - Error Type: {error_type_str}")
        logger.info(f"  - Severity: {result.get('severity')}")
        logger.info(f"  - Pattern Matched: {result.get('pattern_matched')}")
        if result.get("needs_new_pattern"):
            logger.warning(f"  - NEW PATTERN NEEDED for UNKNOWN error")

        # Save complete step result with all sub-agent data
        storage = get_workflow_storage()
        storage.save_step(
            workflow_id=state["workflow_id"],
            step_name="error_analyzer",
            data={
                "error_analysis": error_analysis,
                "error_type": error_type_str,
                "severity": result.get("severity"),
                "pattern_matched": result.get("pattern_matched"),
                "pattern_result": result.get("pattern_result", {}),
                "severity_result": result.get("severity_result", {}),
                "root_cause_result": result.get("root_cause_result", {}),
                "new_pattern_needed": result.get("new_pattern_needed"),
                "needs_new_pattern": result.get("needs_new_pattern", False)
            }
        )

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
        llm_kwargs = {
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "api_key": settings.openai.api_key
        }
        if settings.openai.base_url:
            llm_kwargs["base_url"] = settings.openai.base_url
        llm = ChatOpenAI(**llm_kwargs)

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

        # Extract results
        messages = result.get("messages", [])
        sop_results_text = messages[-1].content if messages else "No SOP results"

        logger.info("SOP search completed")

        # Save step result
        storage = get_workflow_storage()
        storage.save_step(
            workflow_id=state["workflow_id"],
            step_name="sop_searcher",
            data={
                "error_type": error_type,
                "error_message": error_msg,
                "sop_results": sop_results_text
            }
        )

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
        llm_kwargs = {
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "api_key": settings.openai.api_key
        }
        if settings.openai.base_url:
            llm_kwargs["base_url"] = settings.openai.base_url
        llm = ChatOpenAI(**llm_kwargs)

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

        # Extract results
        messages = result.get("messages", [])
        collected_data_text = messages[-1].content if messages else "No data collected"

        logger.info("Data collection completed")

        # Save step result
        storage = get_workflow_storage()
        storage.save_step(
            workflow_id=state["workflow_id"],
            step_name="data_collector",
            data={
                "error_type": error_type,
                "log_file": state['log_file_path'],
                "collected_data": collected_data_text
            }
        )

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
        llm_kwargs = {
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "api_key": settings.openai.api_key
        }
        if settings.openai.base_url:
            llm_kwargs["base_url"] = settings.openai.base_url
        llm = ChatOpenAI(**llm_kwargs)

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

        # Extract decision result
        messages = result.get("messages", [])
        decision_text = messages[-1].content if messages else "No decision"

        # Save step result
        storage = get_workflow_storage()
        storage.save_step(
            workflow_id=state["workflow_id"],
            step_name="decision_maker",
            data={
                "resolution_plan": resolution_plan,
                "decision_text": decision_text
            }
        )

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
        llm_kwargs = {
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "api_key": settings.openai.api_key
        }
        if settings.openai.base_url:
            llm_kwargs["base_url"] = settings.openai.base_url
        llm = ChatOpenAI(**llm_kwargs)

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

        # Extract execution result
        messages = result.get("messages", [])
        execution_text = messages[-1].content if messages else "No execution result"

        logger.info("Auto execution completed")

        # Save step result
        storage = get_workflow_storage()
        storage.save_step(
            workflow_id=state["workflow_id"],
            step_name="auto_executor",
            data={
                "resolution_plan": resolution_plan,
                "execution_result": execution_text,
                "status": "completed"
            }
        )

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
        llm_kwargs = {
            "model": settings.openai.model,
            "temperature": settings.openai.temperature,
            "api_key": settings.openai.api_key
        }
        if settings.openai.base_url:
            llm_kwargs["base_url"] = settings.openai.base_url
        llm = ChatOpenAI(**llm_kwargs)

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

        # Extract notification result
        messages = result.get("messages", [])
        notification_text = messages[-1].content if messages else "No notification"

        logger.info("Notification sent")

        # Save step result
        storage = get_workflow_storage()
        storage.save_step(
            workflow_id=state["workflow_id"],
            step_name="notification",
            data={
                "notification": notification_text,
                "workflow_id": state['workflow_id']
            }
        )

        # Mark workflow as completed
        state["workflow_status"] = "COMPLETED"
        state["completed_at"] = datetime.now().isoformat()
        state["next_agent"] = "END"

        # Save complete workflow result
        storage.save_complete(
            workflow_id=state["workflow_id"],
            final_state=dict(state)
        )

        return state

    except Exception as e:
        logger.error(f"Notification failed: {e}", exc_info=True)
        state["errors"].append(f"Notification failed: {str(e)}")
        state["workflow_status"] = "FAILED"
        state["next_agent"] = "END"
        return state


def decide_next_after_decision_maker(state: AgentState) -> str:
    """
    Decide next node after decision maker.

    의사결정 후 다음 노드를 결정합니다.

    Args:
        state: Current agent state

    Returns:
        str: Next node name ("auto_executor" or "notification")
    """
    resolution_plan = state.get("resolution_plan")
    if resolution_plan and resolution_plan.get("auto_executable"):
        return "auto_executor"
    return "notification"


def create_workflow() -> StateGraph:
    """
    Create LangGraph workflow.

    LangGraph 워크플로우를 생성합니다.

    Returns:
        StateGraph: Configured workflow graph
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
        decide_next_after_decision_maker
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

        # Get workflow storage path
        storage = get_workflow_storage()
        workflow_dir = storage._get_workflow_dir(final_state["workflow_id"])

        logger.info(f"Workflow results saved to: {workflow_dir}")

        return {
            "status": "success",
            "workflow_id": final_state["workflow_id"],
            "workflow_status": final_state["workflow_status"],
            "workflow_dir": str(workflow_dir),
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
        logger.info(f"Results saved to: {result.get('workflow_dir', 'N/A')}")
        logger.info("")
        logger.info("View results:")
        logger.info(f"  - Summary: {result.get('workflow_dir')}/summary.txt")
        logger.info(f"  - Complete: {result.get('workflow_dir')}/complete.json")
        logger.info(f"  - Step files: {result.get('workflow_dir')}/*.json")
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
