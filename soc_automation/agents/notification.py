"""
Notification Agent for SOC automation system.

알림 Agent
- 에러 감지 및 해결 상태 알림
- DB INSERT를 통한 알림 전송
- 우선순위 기반 알림
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..config.agent_prompts import get_agent_prompt
from ..utils.state import NotificationData
from ..utils.logger import get_agent_logger


# Tools for Notification Agent
@tool
def generate_notification_message(
    error_type: str,
    severity: int,
    status: str,
    summary: str
) -> str:
    """
    Generate notification message.

    알림 메시지를 생성합니다.

    Args:
        error_type: Type of error
        severity: Error severity (0-10)
        status: Current status (DETECTED/IN_PROGRESS/RESOLVED/FAILED)
        summary: Error summary

    Returns:
        str: Formatted notification message
    """
    try:
        # Determine priority
        if severity >= 9:
            priority = "CRITICAL"
            emoji = "🔴"
        elif severity >= 7:
            priority = "HIGH"
            emoji = "🟠"
        elif severity >= 4:
            priority = "MEDIUM"
            emoji = "🟡"
        else:
            priority = "LOW"
            emoji = "🟢"

        # Generate subject
        subject = f"[{priority}] SOC Error {status}: {error_type}"

        # Generate body
        body = f"""
{emoji} **SOC Automation Alert**

**Error Type:** {error_type}
**Severity:** {severity}/10
**Status:** {status}
**Priority:** {priority}

**Summary:**
{summary}

**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
This is an automated notification from SOC Automation System.
"""

        return f"""
Generated Notification:

Subject: {subject}
Priority: {priority}
Body:
{body}
"""

    except Exception as e:
        return f"Error generating message: {str(e)}"


@tool
def generate_insert_query(
    notification_id: str,
    severity: int,
    error_type: str,
    status: str,
    message: str,
    table_name: str = "SOC_NOTIFICATIONS"
) -> str:
    """
    Generate SQL INSERT query for notification.

    알림을 위한 SQL INSERT 쿼리를 생성합니다.

    Args:
        notification_id: Unique notification ID
        severity: Error severity
        error_type: Error type
        status: Current status
        message: Notification message
        table_name: Target table name

    Returns:
        str: SQL INSERT query
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Escape single quotes in message
        message_escaped = message.replace("'", "''")

        # Determine priority
        if severity >= 9:
            priority = "CRITICAL"
        elif severity >= 7:
            priority = "HIGH"
        elif severity >= 4:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        query = f"""
INSERT INTO {table_name} (
    notification_id,
    timestamp,
    severity,
    error_type,
    status,
    priority,
    message,
    assigned_to,
    created_at
) VALUES (
    '{notification_id}',
    TO_TIMESTAMP('{timestamp}', 'YYYY-MM-DD HH24:MI:SS'),
    {severity},
    '{error_type}',
    '{status}',
    '{priority}',
    '{message_escaped}',
    'SOC_TEAM',
    SYSDATE
);
"""

        return f"Generated SQL INSERT Query:\n{query}"

    except Exception as e:
        return f"Error generating query: {str(e)}"


@tool
def format_notification_metadata(
    workflow_id: str,
    error_details: str,
    resolution_summary: str = "N/A",
    execution_result: str = "N/A"
) -> str:
    """
    Format notification metadata as JSON.

    알림 메타데이터를 JSON으로 포맷합니다.

    Args:
        workflow_id: Workflow ID
        error_details: Error details
        resolution_summary: Resolution summary
        execution_result: Execution result

    Returns:
        str: JSON metadata
    """
    try:
        metadata = {
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat(),
            "error_details": error_details,
            "resolution_summary": resolution_summary,
            "execution_result": execution_result,
            "automation_system": "SOC Multi-Agent Automation v1.0"
        }

        return json.dumps(metadata, indent=2)

    except Exception as e:
        return f"Error formatting metadata: {str(e)}"


@tool
def determine_recipients(severity: int, error_type: str) -> str:
    """
    Determine notification recipients based on severity and error type.

    심각도와 에러 타입에 따라 알림 수신자를 결정합니다.

    Args:
        severity: Error severity (0-10)
        error_type: Error type

    Returns:
        str: Comma-separated list of recipients
    """
    try:
        recipients = ["soc_team@company.com"]

        # Add management for critical errors
        if severity >= 9:
            recipients.append("soc_manager@company.com")
            recipients.append("engineering_manager@company.com")

        # Add specialists for specific error types
        if error_type in ["MEMORY", "TIMEOUT"]:
            recipients.append("performance_team@company.com")
        elif error_type in ["PROTOCOL", "ASSERTION"]:
            recipients.append("design_team@company.com")
        elif error_type in ["CONFIG", "COMPILATION"]:
            recipients.append("tools_team@company.com")

        return f"Recipients: {', '.join(recipients)}"

    except Exception as e:
        return f"Error determining recipients: {str(e)}"


# Default tools for notification agent
DEFAULT_TOOLS = [
    generate_notification_message,
    generate_insert_query,
    format_notification_metadata,
    determine_recipients,
]


def create_notification_agent(llm, tools: Optional[List] = None):
    """
    Create Notification Agent.

    알림 Agent를 생성합니다.

    Args:
        llm: Language model to use
        tools: List of tools (default: DEFAULT_TOOLS)

    Returns:
        Agent: Notification Agent
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=get_agent_prompt("notification")
    )

    return agent


async def run_notification_agent(
    llm,
    workflow_id: str,
    error_analysis: Dict[str, Any],
    resolution_plan: Optional[Dict[str, Any]] = None,
    execution_result: Optional[Dict[str, Any]] = None,
    tools: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run notification agent.

    알림 Agent를 실행합니다.

    Args:
        llm: Language model
        workflow_id: Workflow ID
        error_analysis: Error analysis results
        resolution_plan: Resolution plan (optional)
        execution_result: Execution result (optional)
        tools: Optional tools list

    Returns:
        Dict: Notification data
    """
    logger = get_agent_logger("notification")
    logger.info(f"Generating notification for workflow: {workflow_id}")

    try:
        # Create agent
        agent = create_notification_agent(llm, tools)

        # Determine status
        if execution_result:
            if execution_result.get("status") == "success":
                status = "RESOLVED"
            elif execution_result.get("status") == "error":
                status = "FAILED"
            else:
                status = "IN_PROGRESS"
        elif resolution_plan:
            status = "IN_PROGRESS"
        else:
            status = "DETECTED"

        # Prepare input message
        input_message = f"""
Please generate a notification for this SOC automation workflow:

Workflow ID: {workflow_id}

=== Error Analysis ===
{json.dumps(error_analysis, indent=2)}

=== Resolution Plan ===
{json.dumps(resolution_plan, indent=2) if resolution_plan else "N/A"}

=== Execution Result ===
{json.dumps(execution_result, indent=2) if execution_result else "N/A"}

Current Status: {status}

Tasks:
1. Generate appropriate notification message based on severity and status
2. Determine notification priority (LOW/MEDIUM/HIGH/CRITICAL)
3. Identify recipients based on severity and error type
4. Generate SQL INSERT query for SOC_NOTIFICATIONS table
5. Format metadata as JSON

Provide a structured notification with:
- notification_type: Type of notification (ERROR_DETECTED/RESOLUTION_STATUS/CRITICAL_ALERT)
- priority: Priority level
- recipients: Target recipients
- subject: Notification subject
- body: Notification body
- db_insert_query: SQL INSERT statement
- metadata: Additional structured data as JSON
"""

        # Run agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]}
        )

        logger.info("Notification generated successfully")

        # Extract results from agent output
        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else "No notification generated"

        return {
            "status": "success",
            "workflow_id": workflow_id,
            "notification_status": status,
            "notification_data": final_message,
            "raw_result": result
        }

    except Exception as e:
        logger.error(f"Notification generation failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "workflow_id": workflow_id,
            "error": str(e)
        }


if __name__ == "__main__":
    """
    Test Notification Agent independently.

    독립적으로 Notification Agent를 테스트합니다.
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Notification Agent Test ===\n")

    # Sample data
    workflow_id = "workflow-12345"

    error_analysis = {
        "error_type": "TIMEOUT",
        "severity": 8,
        "error_message": "Simulation timeout after 3600 seconds",
        "location": "testbench.sv:145"
    }

    resolution_plan = {
        "root_cause": "Infinite loop in DUT",
        "auto_executable": True,
        "resolution_steps": [
            "Increase timeout to 7200",
            "Rerun simulation"
        ]
    }

    execution_result = {
        "status": "success",
        "actions_taken": [
            "Created backup",
            "Modified timeout parameter",
            "Reran simulation"
        ],
        "execution_time": 45.2
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

    # Run notification agent
    async def test():
        result = await run_notification_agent(
            llm=llm,
            workflow_id=workflow_id,
            error_analysis=error_analysis,
            resolution_plan=resolution_plan,
            execution_result=execution_result
        )

        print("\n=== Notification Result ===\n")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Workflow ID: {result['workflow_id']}")
            print(f"Notification Status: {result['notification_status']}")
            print(f"\nNotification Data:\n{result['notification_data']}")
        else:
            print(f"Error: {result.get('error')}")

    # Run test
    asyncio.run(test())

    print("\nTest completed.")
