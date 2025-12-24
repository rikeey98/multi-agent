"""
LangGraph state definitions for SOC automation system.

LangGraph State 정의
- 전체 워크플로우 상태 관리
- Agent 간 데이터 공유
- 실행 진행 상황 추적
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
from enum import Enum
import operator


class ErrorCategory(str, Enum):
    """Error category enumeration"""
    TIMEOUT = "TIMEOUT"
    MEMORY = "MEMORY"
    CONFIG = "CONFIG"
    ASSERTION = "ASSERTION"
    PROTOCOL = "PROTOCOL"
    COMPILATION = "COMPILATION"
    RUNTIME = "RUNTIME"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ExecutionStatus(str, Enum):
    """Execution status enumeration"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"
    SKIPPED = "SKIPPED"


class ErrorAnalysis(TypedDict, total=False):
    """Error analysis result structure"""
    error_type: ErrorCategory
    severity: int  # 0-10
    error_message: str
    location: str  # file:line
    timestamp: str
    context: str
    pattern_match: Optional[str]
    root_cause_hypothesis: str
    affected_modules: List[str]


class SOPResult(TypedDict, total=False):
    """SOP search result structure"""
    sop_id: str
    title: str
    relevance_score: float  # 0-1
    resolution_steps: List[str]
    prerequisites: List[str]
    estimated_time: str
    automation_feasible: bool
    warnings: List[str]
    sop_content: str


class CollectedData(TypedDict, total=False):
    """Collected data structure"""
    logs: Dict[str, str]
    db_records: List[Dict[str, Any]]
    system_status: Dict[str, Any]
    config_files: Dict[str, str]
    changes: List[Dict[str, Any]]
    historical_errors: List[Dict[str, Any]]


class ResolutionPlan(TypedDict, total=False):
    """Resolution plan structure"""
    root_cause: str
    confidence: float  # 0-1
    resolution_steps: List[str]
    auto_executable: bool
    risk_level: RiskLevel
    required_approvals: List[str]
    rollback_plan: List[str]
    expected_outcome: str
    estimated_duration: str


class ExecutionResult(TypedDict, total=False):
    """Execution result structure"""
    execution_status: ExecutionStatus
    actions_taken: List[str]
    backup_location: Optional[str]
    execution_time: float
    verification_result: bool
    rollback_available: bool
    errors: List[str]
    output_logs: str


class NotificationData(TypedDict, total=False):
    """Notification data structure"""
    notification_type: str
    priority: str
    recipients: List[str]
    subject: str
    body: str
    db_insert_query: str
    metadata: Dict[str, Any]
    sent_at: str


class AgentState(TypedDict):
    """
    Main state for LangGraph workflow.

    전체 워크플로우의 상태를 관리합니다.
    Agent 간 정보 공유 및 실행 진행 상황을 추적합니다.
    """
    # Input
    log_file_path: str
    trigger_event: str
    initial_timestamp: str

    # Error Analysis (from error_analyzer)
    error_analysis: Optional[ErrorAnalysis]

    # SOP Search Results (from sop_searcher)
    sop_results: Annotated[List[SOPResult], operator.add]

    # Collected Data (from data_collector)
    collected_data: Optional[CollectedData]

    # Resolution Plan (from decision_maker)
    resolution_plan: Optional[ResolutionPlan]

    # Execution Results (from auto_executor)
    execution_result: Optional[ExecutionResult]

    # Notification (from notification agent)
    notification: Optional[NotificationData]

    # Workflow control
    next_agent: str
    workflow_status: str  # RUNNING/COMPLETED/FAILED
    errors: Annotated[List[str], operator.add]

    # Metadata
    workflow_id: str
    started_at: str
    completed_at: Optional[str]


def create_initial_state(
    log_file_path: str,
    trigger_event: str = "manual",
    workflow_id: Optional[str] = None
) -> AgentState:
    """
    Create initial state for workflow.

    워크플로우의 초기 상태를 생성합니다.

    Args:
        log_file_path: Path to the log file to analyze
        trigger_event: Event that triggered the workflow
        workflow_id: Optional workflow ID (auto-generated if not provided)

    Returns:
        AgentState: Initial state
    """
    from uuid import uuid4

    if workflow_id is None:
        workflow_id = str(uuid4())

    return AgentState(
        # Input
        log_file_path=log_file_path,
        trigger_event=trigger_event,
        initial_timestamp=datetime.now().isoformat(),

        # Agent results
        error_analysis=None,
        sop_results=[],
        collected_data=None,
        resolution_plan=None,
        execution_result=None,
        notification=None,

        # Workflow control
        next_agent="error_analyzer",
        workflow_status="RUNNING",
        errors=[],

        # Metadata
        workflow_id=workflow_id,
        started_at=datetime.now().isoformat(),
        completed_at=None
    )


def update_state_with_error_analysis(
    state: AgentState,
    error_analysis: ErrorAnalysis
) -> AgentState:
    """
    Update state with error analysis results.

    에러 분석 결과로 상태를 업데이트합니다.

    Args:
        state: Current state
        error_analysis: Error analysis result

    Returns:
        AgentState: Updated state
    """
    state["error_analysis"] = error_analysis
    state["next_agent"] = "parallel_collection"  # Next: SOP search + Data collection
    return state


def update_state_with_sop_results(
    state: AgentState,
    sop_results: List[SOPResult]
) -> AgentState:
    """
    Update state with SOP search results.

    SOP 검색 결과로 상태를 업데이트합니다.

    Args:
        state: Current state
        sop_results: SOP search results

    Returns:
        AgentState: Updated state
    """
    state["sop_results"] = sop_results
    return state


def update_state_with_collected_data(
    state: AgentState,
    collected_data: CollectedData
) -> AgentState:
    """
    Update state with collected data.

    수집된 데이터로 상태를 업데이트합니다.

    Args:
        state: Current state
        collected_data: Collected data

    Returns:
        AgentState: Updated state
    """
    state["collected_data"] = collected_data
    return state


def update_state_with_resolution_plan(
    state: AgentState,
    resolution_plan: ResolutionPlan
) -> AgentState:
    """
    Update state with resolution plan.

    해결 계획으로 상태를 업데이트합니다.

    Args:
        state: Current state
        resolution_plan: Resolution plan

    Returns:
        AgentState: Updated state
    """
    state["resolution_plan"] = resolution_plan

    # Determine next agent based on auto_executable flag
    if resolution_plan.get("auto_executable", False):
        state["next_agent"] = "auto_executor"
    else:
        state["next_agent"] = "notification"

    return state


def update_state_with_execution_result(
    state: AgentState,
    execution_result: ExecutionResult
) -> AgentState:
    """
    Update state with execution result.

    실행 결과로 상태를 업데이트합니다.

    Args:
        state: Current state
        execution_result: Execution result

    Returns:
        AgentState: Updated state
    """
    state["execution_result"] = execution_result
    state["next_agent"] = "notification"
    return state


def update_state_with_notification(
    state: AgentState,
    notification: NotificationData
) -> AgentState:
    """
    Update state with notification data.

    알림 데이터로 상태를 업데이트합니다.

    Args:
        state: Current state
        notification: Notification data

    Returns:
        AgentState: Updated state
    """
    state["notification"] = notification
    state["workflow_status"] = "COMPLETED"
    state["completed_at"] = datetime.now().isoformat()
    state["next_agent"] = "END"
    return state


def add_error_to_state(state: AgentState, error: str) -> AgentState:
    """
    Add error to state.

    상태에 에러를 추가합니다.

    Args:
        state: Current state
        error: Error message

    Returns:
        AgentState: Updated state
    """
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append(error)
    return state


if __name__ == "__main__":
    """Test state creation and updates"""
    print("=== Testing AgentState ===\n")

    # Create initial state
    initial_state = create_initial_state(
        log_file_path="/var/log/sim/simulation.log",
        trigger_event="file_watcher"
    )

    print("Initial State:")
    print(f"  Workflow ID: {initial_state['workflow_id']}")
    print(f"  Log File: {initial_state['log_file_path']}")
    print(f"  Next Agent: {initial_state['next_agent']}")
    print(f"  Workflow Status: {initial_state['workflow_status']}")
    print()

    # Simulate error analysis
    error_analysis = ErrorAnalysis(
        error_type=ErrorCategory.TIMEOUT,
        severity=7,
        error_message="Simulation timeout after 3600 seconds",
        location="testbench.sv:145",
        timestamp="2024-01-15T14:30:00",
        context="Running test_case_01",
        pattern_match="TIMEOUT_001",
        root_cause_hypothesis="Infinite loop in DUT",
        affected_modules=["core", "memory_controller"]
    )

    updated_state = update_state_with_error_analysis(initial_state, error_analysis)
    print("After Error Analysis:")
    print(f"  Error Type: {updated_state['error_analysis']['error_type']}")
    print(f"  Severity: {updated_state['error_analysis']['severity']}")
    print(f"  Next Agent: {updated_state['next_agent']}")
    print()

    # Simulate resolution plan
    resolution_plan = ResolutionPlan(
        root_cause="Infinite loop due to clock gating issue",
        confidence=0.85,
        resolution_steps=[
            "Review clock gating logic",
            "Update testbench timeout",
            "Re-run simulation"
        ],
        auto_executable=True,
        risk_level=RiskLevel.LOW,
        required_approvals=[],
        rollback_plan=["Revert configuration changes"],
        expected_outcome="Simulation completes successfully",
        estimated_duration="30 minutes"
    )

    updated_state = update_state_with_resolution_plan(updated_state, resolution_plan)
    print("After Decision Making:")
    print(f"  Auto Executable: {updated_state['resolution_plan']['auto_executable']}")
    print(f"  Risk Level: {updated_state['resolution_plan']['risk_level']}")
    print(f"  Next Agent: {updated_state['next_agent']}")
    print()

    print("State structure validated successfully!")
