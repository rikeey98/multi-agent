"""
Error Analyzer Main Agent.

3개 Sub-Agent 순차 실행:
1. Pattern Matcher - 100+ 패턴 매칭
2. Severity Assessor - 심각도 재평가
3. Root Cause Analyzer - 근본 원인 분석
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.tools import tool

from soc_automation.utils.state import ErrorAnalysis, ErrorCategory
from soc_automation.utils.error_patterns import match_error_pattern, classify_error_by_keywords
from soc_automation.utils.logger import get_agent_logger

from .sub_agents.pattern_matcher import run_pattern_matcher
from .sub_agents.severity_assessor import run_severity_assessor
from .sub_agents.root_cause_analyzer import run_root_cause_analyzer

logger = get_agent_logger("error_analyzer")


# Tools for Error Analyzer
@tool
def read_log_file(file_path: str, num_lines: int = 100) -> str:
    """
    Read log file contents.

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

    Args:
        file_path: Path to log file
        error_keywords: List of keywords to search

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

    Args:
        error_message: Error message to match

    Returns:
        str: Pattern match results
    """
    try:
        result = match_error_pattern(error_message)
        if result:
            return f"Pattern matched: {result}"
        else:
            category = classify_error_by_keywords(error_message)
            return f"No exact pattern match. Classified as: {category}"
    except Exception as e:
        return f"Error matching pattern: {str(e)}"


@tool
def extract_context_from_log(file_path: str, error_line_num: int, context_lines: int = 10) -> str:
    """
    Extract context around an error line from log file.

    Args:
        file_path: Path to log file
        error_line_num: Line number of the error
        context_lines: Number of lines before/after to include

    Returns:
        str: Context around the error
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        start = max(0, error_line_num - context_lines - 1)
        end = min(len(lines), error_line_num + context_lines)

        context = lines[start:end]
        result = f"Context around line {error_line_num}:\n"
        for i, line in enumerate(context, start=start+1):
            prefix = ">>> " if i == error_line_num else "    "
            result += f"{prefix}{i}: {line}"

        return result

    except Exception as e:
        return f"Error extracting context: {str(e)}"


# Default tools
DEFAULT_TOOLS = [
    read_log_file,
    search_error_in_log,
    match_error_patterns_tool,
    extract_context_from_log,
]


def _extract_component(error_message: str) -> str:
    """Extract component name from error message."""
    # Simple heuristic: look for common patterns
    patterns = [
        r'in\s+(\w+(?:_\w+)*)',  # "in module_name"
        r'at\s+(\w+(?:_\w+)*)',  # "at component_name"
        r'(\w+(?:_\w+)*)\s+error', # "component_name error"
    ]

    for pattern in patterns:
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            return match.group(1)

    return "unknown"


async def run_error_analyzer(
    llm,
    log_file_path: str,
    tools: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run error analyzer with 3 sub-agents in sequence.

    순차 실행:
    1. Pattern Matcher → 100+ 패턴 매칭
    2. Severity Assessor → 심각도 재평가
    3. Root Cause Analyzer → 근본 원인 분석

    Args:
        llm: Language model
        log_file_path: Path to log file
        tools: Optional tools list (not used by sub-agents)

    Returns:
        Dict: Error analysis results
    """
    logger.info(f"Starting error analysis for: {log_file_path}")

    try:
        # Read log file
        log_content = read_log_file.invoke({"file_path": log_file_path, "num_lines": 200})

        # Extract error message (simplified - take first error line)
        error_message = ""
        for line in log_content.split('\n'):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['error', 'fatal', 'fail', 'timeout', 'violation']):
                error_message = line.strip()
                break

        if not error_message:
            error_message = "No clear error message found in log"

        logger.info(f"Extracted error message: {error_message[:100]}...")

        # Step 1: Pattern Matching
        logger.info("Step 1/3: Running Pattern Matcher...")
        pattern_result = await run_pattern_matcher(
            llm=llm,
            error_message=error_message,
            log_context=log_content[:500]  # First 500 chars of log
        )

        # Determine error type and base severity
        if pattern_result['matched']:
            error_type = pattern_result['pattern_code']
            base_severity = pattern_result['base_severity']
            pattern_matched = True
            logger.info(f"Pattern matched: {error_type} (severity: {base_severity})")
        else:
            error_type = "UNKNOWN"
            base_severity = 5  # Default for unknown
            pattern_matched = False
            logger.warning(f"No pattern matched. Consider adding new pattern.")

        # Step 2: Severity Assessment
        logger.info("Step 2/3: Running Severity Assessor...")
        component = _extract_component(error_message)
        severity_result = await run_severity_assessor(
            llm=llm,
            error_message=error_message,
            base_severity=base_severity,
            error_type=error_type,
            component=component
        )

        final_severity = severity_result['final_severity']
        logger.info(f"Severity assessment: {base_severity} → {final_severity}")

        # Step 3: Root Cause Analysis
        logger.info("Step 3/3: Running Root Cause Analyzer...")
        root_cause_result = await run_root_cause_analyzer(
            llm=llm,
            error_message=error_message,
            error_type=error_type,
            severity=final_severity
        )

        logger.info(f"Root cause analysis completed. Confidence: {root_cause_result['confidence']}")

        # Build final result
        result = {
            "status": "success",
            "error_type": error_type,
            "severity": final_severity,
            "pattern_matched": pattern_matched,
            "error_message": error_message,
            "component": component,
            "pattern_result": pattern_result,
            "severity_result": severity_result,
            "root_cause_result": root_cause_result,
            "needs_new_pattern": root_cause_result.get('needs_new_pattern', False),
        }

        # Flag for UNKNOWN errors
        if not pattern_matched:
            result["new_pattern_needed"] = {
                "error_message": error_message,
                "suggested_analysis": root_cause_result.get('suggested_pattern', {})
            }

        return result

    except Exception as e:
        logger.error(f"Error analyzer failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "error_type": "UNKNOWN",
            "severity": 5
        }


def create_error_analyzer_agent(llm, tools: Optional[List] = None):
    """
    Create error analyzer agent (for compatibility).

    Note: The new error analyzer uses sub-agents and doesn't
    create a traditional agent. Use run_error_analyzer() directly.
    """
    # Return None or a placeholder
    # The actual execution is done via run_error_analyzer()
    logger.warning("create_error_analyzer_agent is deprecated. Use run_error_analyzer() directly.")
    return None
