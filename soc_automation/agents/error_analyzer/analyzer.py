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


if __name__ == "__main__":
    """
    Test Error Analyzer independently with sub-agents.

    독립적으로 Error Analyzer를 테스트합니다 (3개 Sub-Agent 사용).
    """
    import os
    import sys
    import asyncio
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    load_dotenv()

    print("=== Error Analyzer Test (with Sub-Agents) ===\n")

    # Create sample log file
    log_dir = Path("/tmp/test_logs")
    log_dir.mkdir(exist_ok=True)

    # Sample log with cache coherency error (MEM-001)
    sample_log = """[2024-12-28 10:15:23] INFO: Simulation started
[2024-12-28 10:15:24] INFO: Clock frequency: 100 MHz
[2024-12-28 10:15:25] INFO: Loading testbench...
[2024-12-28 10:15:26] INFO: Initializing DUT...
[2024-12-28 10:15:30] INFO: Running test case 1...
[2024-12-28 10:16:45] ERROR: Cache coherency violation detected at address 0x1000
[2024-12-28 10:16:45] ERROR: Snoop conflict in cache_controller module
[2024-12-28 10:16:45] ERROR: Expected MESI state: Modified, Actual state: Shared
[2024-12-28 10:16:45] FATAL: Simulation stopped due to cache coherency violation
[2024-12-28 10:16:46] INFO: Simulation ended with errors
"""

    log_file = log_dir / "cache_error.log"
    with open(log_file, 'w') as f:
        f.write(sample_log)

    print(f"Created sample log: {log_file}\n")

    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        print("Please set OPENAI_API_KEY in .env file")
        exit(1)

    llm_kwargs = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.7
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**llm_kwargs)

    # Run error analyzer
    async def test():
        result = await run_error_analyzer(
            llm=llm,
            log_file_path=str(log_file)
        )

        print("\n=== Error Analysis Result ===\n")
        print(f"Status: {result['status']}")

        if result['status'] == 'success':
            print(f"\n📋 Error Type: {result['error_type']}")
            print(f"⚠️  Severity: {result['severity']}/10")
            print(f"✅ Pattern Matched: {result['pattern_matched']}")
            print(f"\n📝 Error Message:")
            print(f"   {result['error_message'][:200]}...")

            if result.get('pattern_matched'):
                pattern = result.get('pattern_result', {})
                print(f"\n🎯 Pattern Match:")
                print(f"   - Code: {pattern.get('pattern_code')}")
                print(f"   - Name: {pattern.get('pattern_name')}")
                print(f"   - Confidence: {pattern.get('confidence')}")

                # Display RAG information
                print(f"\n🔍 RAG (Retrieval-Augmented Generation):")
                print(f"   - RAG Used: {pattern.get('rag_used', False)}")
                if pattern.get('rag_used'):
                    print(f"   - Query: {pattern.get('rag_query', 'N/A')[:80]}...")
                    print(f"   - Retrieved Documents: {len(pattern.get('rag_results', []))}")
                    if pattern.get('rag_results'):
                        print(f"   - Top Retrieved Sources:")
                        for idx, doc in enumerate(pattern['rag_results'][:3], 1):
                            content = doc.get('content', '')[:100]
                            print(f"     {idx}. {content}...")
                else:
                    print(f"   - Used LLM-only matching (no RAG)")

            severity = result.get('severity_result', {})
            print(f"\n📊 Severity Assessment:")
            print(f"   - Base Severity: {severity.get('base_severity')}")
            print(f"   - Final Severity: {severity.get('final_severity')}")
            if severity.get('modifiers'):
                print(f"   - Modifiers:")
                for mod in severity['modifiers']:
                    print(f"     * {mod.get('type')}: {mod.get('value'):+d} - {mod.get('reason')}")

            root_cause = result.get('root_cause_result', {})
            print(f"\n🔍 Root Cause Analysis:")
            print(f"   - Hypothesis: {root_cause.get('hypothesis', 'N/A')[:150]}...")
            print(f"   - Confidence: {root_cause.get('confidence')}")
            if root_cause.get('recommended_actions'):
                print(f"   - Recommended Actions:")
                for i, action in enumerate(root_cause['recommended_actions'][:3], 1):
                    print(f"     {i}. {action}")

            if result.get('needs_new_pattern'):
                print(f"\n⚠️  NEW PATTERN NEEDED - Unknown error detected")
                new_pattern = result.get('new_pattern_needed', {})
                print(f"   Suggested pattern should be added to pattern database")
        else:
            print(f"Error: {result.get('error')}")

    # Run test
    asyncio.run(test())

    print(f"\n✅ Test completed. Log file: {log_file}")
