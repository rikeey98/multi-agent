"""
Error Analyzer Agent for SOC automation system.

에러 분석 Agent
- 로그 파일에서 에러 자동 감지
- 에러 분류 및 심각도 판단
- 패턴 매칭 기반 분석
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..config.agent_prompts import get_agent_prompt
from ..utils.state import ErrorAnalysis, ErrorCategory
from ..utils.error_patterns import match_error_pattern, classify_error_by_keywords
from ..utils.logger import get_agent_logger


# Tools for Error Analyzer Agent
@tool
def read_log_file(file_path: str, num_lines: int = 100) -> str:
    """
    Read log file contents.

    로그 파일을 읽습니다.

    Args:
        file_path: Path to log file
        num_lines: Number of lines to read from end (default: 100)

    Returns:
        str: Log file contents
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Return last num_lines
        if len(lines) > num_lines:
            lines = lines[-num_lines:]

        content = ''.join(lines)
        return f"Log file: {file_path}\n{'='*80}\n{content}"

    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def search_error_in_log(file_path: str, error_keywords: List[str]) -> str:
    """
    Search for error keywords in log file.

    로그 파일에서 에러 키워드를 검색합니다.

    Args:
        file_path: Path to log file
        error_keywords: List of keywords to search (e.g., ["ERROR", "FATAL", "timeout"])

    Returns:
        str: Lines containing error keywords
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        error_lines = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line_lower = line.lower()
                for keyword in error_keywords:
                    if keyword.lower() in line_lower:
                        error_lines.append(f"Line {line_num}: {line.strip()}")
                        break

        if not error_lines:
            return "No errors found with the specified keywords."

        result = f"Found {len(error_lines)} error lines:\n" + '\n'.join(error_lines[-50:])
        return result

    except Exception as e:
        return f"Error searching file: {str(e)}"


@tool
def match_error_patterns_tool(error_message: str) -> str:
    """
    Match error message against known error patterns.

    에러 메시지를 알려진 패턴과 매칭합니다.

    Args:
        error_message: Error message to analyze

    Returns:
        str: Pattern match result with category and severity
    """
    result = match_error_pattern(error_message)

    if result:
        pattern, match = result
        return f"""
Pattern Match Found:
- Pattern ID: {pattern.pattern_id}
- Category: {pattern.category.value}
- Severity: {pattern.severity}/10
- Description: {pattern.description}
- Suggested Action: {pattern.suggested_action}
- Matched Text: {match.group(0)}
"""
    else:
        # Try keyword-based classification
        category = classify_error_by_keywords(error_message)
        if category:
            return f"""
No exact pattern match found.
Classified by keywords as: {category.value}
Suggested action: Manual review required.
"""
        else:
            return "No pattern match found. Error category: UNKNOWN"


@tool
def extract_context_from_log(file_path: str, error_line_number: int, context_lines: int = 10) -> str:
    """
    Extract context around an error line.

    에러 라인 주변의 컨텍스트를 추출합니다.

    Args:
        file_path: Path to log file
        error_line_number: Line number where error occurred
        context_lines: Number of context lines before/after (default: 10)

    Returns:
        str: Context lines around the error
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        start = max(0, error_line_number - context_lines - 1)
        end = min(len(lines), error_line_number + context_lines)

        context = []
        for i in range(start, end):
            marker = ">>> " if i == error_line_number - 1 else "    "
            context.append(f"{marker}Line {i+1}: {lines[i].rstrip()}")

        return '\n'.join(context)

    except Exception as e:
        return f"Error extracting context: {str(e)}"


# Default tools for error analyzer
DEFAULT_TOOLS = [
    read_log_file,
    search_error_in_log,
    match_error_patterns_tool,
    extract_context_from_log,
]


def create_error_analyzer_agent(llm, tools: Optional[List] = None):
    """
    Create Error Analyzer Agent.

    에러 분석 Agent를 생성합니다.

    Args:
        llm: Language model to use
        tools: List of tools (default: DEFAULT_TOOLS)

    Returns:
        Agent: Error Analyzer Agent
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=get_agent_prompt("error_analyzer")
    )

    return agent


async def run_error_analyzer(
    llm,
    log_file_path: str,
    tools: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run error analyzer agent.

    에러 분석 Agent를 실행합니다.

    Args:
        llm: Language model
        log_file_path: Path to log file to analyze
        tools: Optional tools list

    Returns:
        Dict: Error analysis result
    """
    logger = get_agent_logger("error_analyzer")
    logger.info(f"Starting error analysis for: {log_file_path}")

    try:
        # Create agent
        agent = create_error_analyzer_agent(llm, tools)

        # Prepare input message
        input_message = f"""
Please analyze the log file: {log_file_path}

Tasks:
1. Read the log file and identify errors
2. Extract key error messages
3. Match errors against known patterns
4. Classify error type and determine severity (0-10)
5. Extract location (file:line if available)
6. Provide root cause hypothesis

Provide a structured analysis with:
- error_type: Category (TIMEOUT/MEMORY/CONFIG/ASSERTION/PROTOCOL/COMPILATION/RUNTIME/UNKNOWN)
- severity: 0-10
- error_message: The actual error message
- location: file:line or "unknown"
- timestamp: timestamp if found, or "unknown"
- context: relevant log context
- pattern_match: pattern ID if matched
- root_cause_hypothesis: your hypothesis about the root cause
"""

        # Run agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]}
        )

        logger.info("Error analysis completed")

        # Extract analysis from agent output
        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else "No analysis"

        return {
            "status": "success",
            "log_file": log_file_path,
            "analysis": final_message,
            "raw_result": result
        }

    except Exception as e:
        logger.error(f"Error analysis failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "log_file": log_file_path,
            "error": str(e)
        }


if __name__ == "__main__":
    """
    Test Error Analyzer Agent independently.

    독립적으로 Error Analyzer Agent를 테스트합니다.
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== Error Analyzer Agent Test ===\n")

    # Create a sample log file for testing
    sample_log_path = "/tmp/test_sim.log"
    sample_log_content = """
[2024-01-15 14:30:00] INFO: Starting simulation
[2024-01-15 14:30:05] INFO: Initializing modules
[2024-01-15 14:30:10] INFO: Running test_case_01
[2024-01-15 14:35:00] WARNING: Simulation running longer than expected
[2024-01-15 15:00:00] ERROR: Simulation timeout after 3600 seconds
[2024-01-15 15:00:00] ERROR: Testbench failed at testbench.sv:145
[2024-01-15 15:00:01] FATAL: Aborting simulation
[2024-01-15 15:00:02] INFO: Simulation stopped
"""

    # Write sample log
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

    # Run error analyzer
    async def test():
        result = await run_error_analyzer(llm, sample_log_path)

        print("\n=== Analysis Result ===\n")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"\nAnalysis:\n{result['analysis']}")
        else:
            print(f"Error: {result.get('error')}")

    # Run test
    asyncio.run(test())

    print(f"\nTest completed. Log file: {sample_log_path}")
