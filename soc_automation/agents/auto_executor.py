"""
Auto Executor Agent for SOC automation system.

자동 실행 Agent
- 승인된 작업을 안전하게 자동 실행
- 백업 및 롤백 지원
- 실행 검증
"""

import asyncio
import subprocess
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import time

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..config.agent_prompts import get_agent_prompt
from ..utils.state import ExecutionResult, ExecutionStatus
from ..utils.logger import get_agent_logger


# Tools for Auto Executor Agent
@tool
def create_backup(file_path: str, backup_dir: str = "/var/backup/soc") -> str:
    """
    Create backup of file before modification.

    파일 수정 전 백업을 생성합니다.

    Args:
        file_path: Path to file to backup
        backup_dir: Backup directory

    Returns:
        str: Backup location or error message
    """
    try:
        source = Path(file_path)
        if not source.exists():
            return f"Error: Source file not found: {file_path}"

        # Create backup directory
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.name}.{timestamp}.backup"
        backup_file = backup_path / backup_name

        # Copy file
        shutil.copy2(source, backup_file)

        return f"Backup created successfully: {backup_file}"

    except Exception as e:
        return f"Error creating backup: {str(e)}"


@tool
def modify_config_file(file_path: str, parameter: str, new_value: str) -> str:
    """
    Modify configuration file parameter.

    설정 파일의 파라미터를 수정합니다.

    Args:
        file_path: Path to config file
        parameter: Parameter name to modify
        new_value: New value for parameter

    Returns:
        str: Modification result
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: Config file not found: {file_path}"

        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find and modify parameter
        modified = False
        for i, line in enumerate(lines):
            if line.strip().startswith(parameter):
                # Preserve format
                if '=' in line:
                    prefix = line.split('=')[0] + '='
                    lines[i] = f"{prefix} {new_value}\n"
                elif ':' in line:
                    prefix = line.split(':')[0] + ':'
                    lines[i] = f"{prefix} {new_value}\n"
                else:
                    lines[i] = f"{parameter} {new_value}\n"
                modified = True
                break

        if not modified:
            return f"Parameter '{parameter}' not found in {file_path}"

        # Write back
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return f"Successfully modified {parameter} = {new_value} in {file_path}"

    except Exception as e:
        return f"Error modifying config: {str(e)}"


@tool
def execute_command(command: str, timeout: int = 300) -> str:
    """
    Execute shell command safely.

    쉘 명령을 안전하게 실행합니다.

    Args:
        command: Command to execute
        timeout: Timeout in seconds (default: 300)

    Returns:
        str: Command output
    """
    try:
        # Safety check - block dangerous commands
        dangerous_patterns = ['rm -rf /', 'dd if=', 'mkfs', ':(){ :|:& };:', 'chmod -R 777 /']
        for pattern in dangerous_patterns:
            if pattern in command:
                return f"Error: Dangerous command blocked: {command}"

        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = f"Command: {command}\n"
        output += f"Return Code: {result.returncode}\n"
        output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"

        return output

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def verify_file_exists(file_path: str) -> str:
    """
    Verify that a file exists.

    파일이 존재하는지 확인합니다.

    Args:
        file_path: Path to file

    Returns:
        str: Verification result
    """
    try:
        path = Path(file_path)
        if path.exists():
            stats = path.stat()
            return f"""
File exists: {file_path}
Size: {stats.st_size} bytes
Modified: {datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            return f"File does not exist: {file_path}"

    except Exception as e:
        return f"Error verifying file: {str(e)}"


@tool
def rollback_from_backup(backup_path: str, target_path: str) -> str:
    """
    Restore file from backup.

    백업에서 파일을 복원합니다.

    Args:
        backup_path: Path to backup file
        target_path: Path to restore to

    Returns:
        str: Rollback result
    """
    try:
        backup = Path(backup_path)
        target = Path(target_path)

        if not backup.exists():
            return f"Error: Backup file not found: {backup_path}"

        # Restore file
        shutil.copy2(backup, target)

        return f"Successfully restored {target_path} from backup {backup_path}"

    except Exception as e:
        return f"Error during rollback: {str(e)}"


# Default tools for auto executor
DEFAULT_TOOLS = [
    create_backup,
    modify_config_file,
    execute_command,
    verify_file_exists,
    rollback_from_backup,
]


def create_auto_executor_agent(llm, tools: Optional[List] = None):
    """
    Create Auto Executor Agent.

    자동 실행 Agent를 생성합니다.

    Args:
        llm: Language model to use
        tools: List of tools (default: DEFAULT_TOOLS)

    Returns:
        Agent: Auto Executor Agent
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=get_agent_prompt("auto_executor")
    )

    return agent


async def run_auto_executor(
    llm,
    resolution_plan: Dict[str, Any],
    tools: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run auto executor agent.

    자동 실행 Agent를 실행합니다.

    Args:
        llm: Language model
        resolution_plan: Resolution plan from decision maker
        tools: Optional tools list

    Returns:
        Dict: Execution result
    """
    logger = get_agent_logger("auto_executor")
    logger.info("Starting auto execution")

    start_time = time.time()

    try:
        # Check if auto-executable
        if not resolution_plan.get("auto_executable", False):
            logger.warning("Resolution plan is not marked as auto-executable")
            return {
                "status": "skipped",
                "reason": "Not approved for auto-execution",
                "execution_time": 0
            }

        # Create agent
        agent = create_auto_executor_agent(llm, tools)

        # Prepare input message
        import json
        input_message = f"""
Please execute the following resolution plan safely:

{json.dumps(resolution_plan, indent=2)}

IMPORTANT SAFETY REQUIREMENTS:
1. Create backups of all files before modification
2. Verify pre-conditions before each action
3. Execute actions in the specified order
4. Verify each action's success before proceeding
5. If any action fails, stop and prepare for rollback

Tasks:
1. Review the resolution steps
2. Create necessary backups
3. Execute each step sequentially
4. Verify execution success
5. Document all actions taken

Provide execution result with:
- execution_status: SUCCESS/FAILURE/PARTIAL
- actions_taken: List of executed actions
- backup_location: Path to backup files
- verification_result: Whether verification passed
- rollback_available: Whether rollback is possible
- errors: Any errors encountered
"""

        # Run agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]}
        )

        execution_time = time.time() - start_time
        logger.info(f"Auto execution completed in {execution_time:.2f}s")

        # Extract results from agent output
        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else "No execution result"

        return {
            "status": "success",
            "execution_result": final_message,
            "execution_time": execution_time,
            "raw_result": result
        }

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Auto execution failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "execution_time": execution_time
        }


if __name__ == "__main__":
    """
    Test Auto Executor Agent independently.

    독립적으로 Auto Executor Agent를 테스트합니다.
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Auto Executor Agent Test ===\n")

    # Create test config file
    test_config = Path("/tmp/test_sim.cfg")
    test_config.write_text("""
# Simulation Configuration
timeout = 3600
memory_limit = 8GB
log_level = INFO
""")

    print(f"Created test config: {test_config}\n")

    # Sample resolution plan
    resolution_plan = {
        "root_cause": "Simulation timeout due to low timeout value",
        "auto_executable": True,
        "risk_level": "LOW",
        "resolution_steps": [
            f"Create backup of {test_config}",
            f"Modify timeout parameter to 7200 in {test_config}",
            "Verify configuration file",
        ],
        "rollback_plan": ["Restore from backup if needed"]
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

    # Run auto executor
    async def test():
        result = await run_auto_executor(
            llm=llm,
            resolution_plan=resolution_plan
        )

        print("\n=== Execution Result ===\n")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Execution Time: {result['execution_time']:.2f}s")
            print(f"\nResult:\n{result['execution_result']}")
        else:
            print(f"Error: {result.get('error')}")

        # Check modified config
        print(f"\n=== Modified Config ===\n")
        print(test_config.read_text())

    # Run test
    asyncio.run(test())

    print("\nTest completed.")
