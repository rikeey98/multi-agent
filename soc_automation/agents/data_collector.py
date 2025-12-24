"""
Data Collector Agent for SOC automation system.

데이터 수집 Agent
- 로그 파일 수집
- 시스템 상태 확인
- DB 조회
- 설정 파일 수집
"""

import asyncio
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..config.agent_prompts import get_agent_prompt
from ..utils.state import CollectedData
from ..utils.logger import get_agent_logger


# Tools for Data Collector Agent
@tool
def collect_system_info() -> str:
    """
    Collect system information (CPU, memory, disk).

    시스템 정보를 수집합니다 (CPU, 메모리, 디스크).

    Returns:
        str: System information
    """
    try:
        info = {}

        # CPU info
        try:
            cpu_info = subprocess.check_output(['lscpu'], text=True)
            cpu_lines = [line for line in cpu_info.split('\n') if 'CPU(s):' in line or 'Model name:' in line]
            info['cpu'] = '\n'.join(cpu_lines[:3])
        except Exception:
            info['cpu'] = "N/A"

        # Memory info
        try:
            mem_info = subprocess.check_output(['free', '-h'], text=True)
            info['memory'] = mem_info
        except Exception:
            info['memory'] = "N/A"

        # Disk info
        try:
            disk_info = subprocess.check_output(['df', '-h'], text=True)
            info['disk'] = disk_info
        except Exception:
            info['disk'] = "N/A"

        result = f"""
System Information:

CPU:
{info['cpu']}

Memory:
{info['memory']}

Disk:
{info['disk']}
"""
        return result

    except Exception as e:
        return f"Error collecting system info: {str(e)}"


@tool
def check_running_processes(process_filter: str = "sim") -> str:
    """
    Check running processes related to simulation.

    시뮬레이션 관련 실행 중인 프로세스를 확인합니다.

    Args:
        process_filter: Filter string for processes (default: "sim")

    Returns:
        str: Running processes information
    """
    try:
        # Get processes
        ps_output = subprocess.check_output(
            ['ps', 'aux'],
            text=True
        )

        # Filter processes
        lines = ps_output.split('\n')
        header = lines[0]
        filtered = [line for line in lines[1:] if process_filter.lower() in line.lower()]

        if not filtered:
            return f"No processes found matching: {process_filter}"

        result = f"Processes matching '{process_filter}':\n\n{header}\n"
        result += '\n'.join(filtered[:20])  # Limit to 20 processes

        return result

    except Exception as e:
        return f"Error checking processes: {str(e)}"


@tool
def read_config_file(config_path: str) -> str:
    """
    Read configuration file.

    설정 파일을 읽습니다.

    Args:
        config_path: Path to configuration file

    Returns:
        str: Configuration file content
    """
    try:
        path = Path(config_path)
        if not path.exists():
            return f"Configuration file not found: {config_path}"

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return f"Configuration File: {path.name}\n{'='*80}\n{content}"

    except Exception as e:
        return f"Error reading config file: {str(e)}"


@tool
def collect_log_excerpts(log_file: str, keywords: List[str], num_lines: int = 50) -> str:
    """
    Collect log excerpts around keywords.

    키워드 주변의 로그 발췌를 수집합니다.

    Args:
        log_file: Path to log file
        keywords: Keywords to search for
        num_lines: Number of context lines around match

    Returns:
        str: Log excerpts
    """
    try:
        path = Path(log_file)
        if not path.exists():
            return f"Log file not found: {log_file}"

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        excerpts = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for keyword in keywords:
                if keyword.lower() in line_lower:
                    # Get context
                    start = max(0, i - 5)
                    end = min(len(lines), i + 6)
                    context = ''.join(lines[start:end])
                    excerpts.append(f"[Match at line {i+1}]\n{context}\n")
                    break

        if not excerpts:
            return f"No matches found for keywords: {keywords}"

        # Limit excerpts
        excerpts = excerpts[:10]  # Max 10 excerpts
        result = f"Log Excerpts (found {len(excerpts)} matches):\n\n"
        result += '\n---\n'.join(excerpts)

        return result

    except Exception as e:
        return f"Error collecting log excerpts: {str(e)}"


@tool
def check_file_changes(directory: str, hours: int = 24) -> str:
    """
    Check for recently modified files.

    최근 수정된 파일을 확인합니다.

    Args:
        directory: Directory to check
        hours: Number of hours to look back (default: 24)

    Returns:
        str: Recently modified files
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Directory not found: {directory}"

        # Find files modified in last N hours
        import time
        cutoff_time = time.time() - (hours * 3600)

        modified_files = []
        for file_path in path.rglob('*'):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if mtime > cutoff_time:
                    mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    modified_files.append({
                        'file': str(file_path),
                        'modified': mtime_str,
                        'size': file_path.stat().st_size
                    })

        if not modified_files:
            return f"No files modified in last {hours} hours"

        # Sort by modification time
        modified_files.sort(key=lambda x: x['modified'], reverse=True)

        result = f"Files modified in last {hours} hours:\n\n"
        for f in modified_files[:20]:  # Limit to 20 files
            result += f"{f['modified']} - {f['file']} ({f['size']} bytes)\n"

        return result

    except Exception as e:
        return f"Error checking file changes: {str(e)}"


@tool
def get_environment_variables(filter_prefix: str = "") -> str:
    """
    Get environment variables.

    환경 변수를 가져옵니다.

    Args:
        filter_prefix: Optional prefix to filter variables (e.g., "SIM_")

    Returns:
        str: Environment variables
    """
    try:
        import os

        if filter_prefix:
            env_vars = {k: v for k, v in os.environ.items() if k.startswith(filter_prefix)}
        else:
            # Return common simulation-related variables
            common_prefixes = ['SIM', 'VCS', 'UVM', 'QUESTA', 'MODEL']
            env_vars = {k: v for k, v in os.environ.items()
                       if any(k.startswith(prefix) for prefix in common_prefixes)}

        if not env_vars:
            return f"No environment variables found with filter: {filter_prefix or 'common prefixes'}"

        result = "Environment Variables:\n\n"
        for key, value in sorted(env_vars.items()):
            result += f"{key}={value}\n"

        return result

    except Exception as e:
        return f"Error getting environment variables: {str(e)}"


# Default tools for data collector
DEFAULT_TOOLS = [
    collect_system_info,
    check_running_processes,
    read_config_file,
    collect_log_excerpts,
    check_file_changes,
    get_environment_variables,
]


def create_data_collector_agent(llm, tools: Optional[List] = None):
    """
    Create Data Collector Agent.

    데이터 수집 Agent를 생성합니다.

    Args:
        llm: Language model to use
        tools: List of tools (default: DEFAULT_TOOLS)

    Returns:
        Agent: Data Collector Agent
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=get_agent_prompt("data_collector")
    )

    return agent


async def run_data_collector(
    llm,
    log_file: str,
    error_type: str,
    config_dir: Optional[str] = None,
    tools: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run data collector agent.

    데이터 수집 Agent를 실행합니다.

    Args:
        llm: Language model
        log_file: Path to log file
        error_type: Type of error
        config_dir: Configuration directory (optional)
        tools: Optional tools list

    Returns:
        Dict: Collected data
    """
    logger = get_agent_logger("data_collector")
    logger.info(f"Starting data collection for error type: {error_type}")

    try:
        # Create agent
        agent = create_data_collector_agent(llm, tools)

        # Prepare input message
        config_info = f"Configuration directory: {config_dir}" if config_dir else "No config directory specified"

        input_message = f"""
Please collect relevant data for this error:

Error Type: {error_type}
Log File: {log_file}
{config_info}

Tasks:
1. Collect system information (CPU, memory, disk)
2. Check running processes related to simulation
3. Collect log excerpts around the error
4. Check for recently modified files
5. Read relevant configuration files
6. Collect environment variables

Focus on gathering data that will help diagnose the {error_type} error.

Provide a structured summary with:
- system_status: Current system state
- logs: Relevant log excerpts
- config_files: Configuration file contents
- changes: Recent file changes
- processes: Running processes
- environment: Relevant environment variables
"""

        # Run agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]}
        )

        logger.info("Data collection completed")

        # Extract results from agent output
        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else "No data collected"

        return {
            "status": "success",
            "error_type": error_type,
            "collected_data": final_message,
            "raw_result": result
        }

    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": error_type,
            "error": str(e)
        }


if __name__ == "__main__":
    """
    Test Data Collector Agent independently.

    독립적으로 Data Collector Agent를 테스트합니다.
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Data Collector Agent Test ===\n")

    # Create sample log file
    sample_log_path = "/tmp/test_sim.log"
    sample_log_content = """
[2024-01-15 14:30:00] INFO: Starting simulation
[2024-01-15 14:30:05] INFO: Initializing modules
[2024-01-15 14:30:10] INFO: Running test_case_01
[2024-01-15 14:35:00] WARNING: Simulation running longer than expected
[2024-01-15 15:00:00] ERROR: Simulation timeout after 3600 seconds
[2024-01-15 15:00:00] ERROR: Testbench failed at testbench.sv:145
"""

    with open(sample_log_path, 'w') as f:
        f.write(sample_log_content)

    print(f"Created sample log file: {sample_log_path}\n")

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

    # Run data collector
    async def test():
        result = await run_data_collector(
            llm=llm,
            log_file=sample_log_path,
            error_type="TIMEOUT"
        )

        print("\n=== Data Collection Result ===\n")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"\nCollected Data:\n{result['collected_data']}")
        else:
            print(f"Error: {result.get('error')}")

    # Run test
    asyncio.run(test())

    print(f"\nTest completed.")
