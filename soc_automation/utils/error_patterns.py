"""
Error pattern definitions for SOC automation system.

에러 패턴 정의
- 정규표현식 기반 에러 패턴 매칭
- 에러 카테고리별 패턴 분류
- 심각도 자동 판단
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .state import ErrorCategory


@dataclass
class ErrorPattern:
    """
    Error pattern definition.

    에러 패턴 정의 클래스
    """
    pattern_id: str
    category: ErrorCategory
    regex: str
    severity: int  # 0-10
    description: str
    keywords: List[str]
    suggested_action: str


# Timeout Error Patterns
TIMEOUT_PATTERNS = [
    ErrorPattern(
        pattern_id="TIMEOUT_001",
        category=ErrorCategory.TIMEOUT,
        regex=r".*timeout.*after\s+(\d+)\s+(seconds|ms|minutes)",
        severity=7,
        description="Simulation timeout",
        keywords=["timeout", "exceeded", "limit"],
        suggested_action="Check for infinite loops or increase timeout value"
    ),
    ErrorPattern(
        pattern_id="TIMEOUT_002",
        category=ErrorCategory.TIMEOUT,
        regex=r".*watchdog.*timeout",
        severity=8,
        description="Watchdog timeout",
        keywords=["watchdog", "timeout"],
        suggested_action="Review watchdog configuration and system responsiveness"
    ),
    ErrorPattern(
        pattern_id="TIMEOUT_003",
        category=ErrorCategory.TIMEOUT,
        regex=r".*hung.*task.*blocked.*for.*seconds",
        severity=7,
        description="Task hung/blocked",
        keywords=["hung", "blocked", "task"],
        suggested_action="Investigate deadlock or resource contention"
    ),
]


# Memory Error Patterns
MEMORY_PATTERNS = [
    ErrorPattern(
        pattern_id="MEMORY_001",
        category=ErrorCategory.MEMORY,
        regex=r".*(out of memory|OOM|memory exhausted)",
        severity=9,
        description="Out of memory",
        keywords=["out of memory", "OOM", "exhausted"],
        suggested_action="Increase memory allocation or optimize memory usage"
    ),
    ErrorPattern(
        pattern_id="MEMORY_002",
        category=ErrorCategory.MEMORY,
        regex=r".*(memory leak|leak detected)",
        severity=8,
        description="Memory leak detected",
        keywords=["leak", "memory leak"],
        suggested_action="Review memory allocation and deallocation"
    ),
    ErrorPattern(
        pattern_id="MEMORY_003",
        category=ErrorCategory.MEMORY,
        regex=r".*(segmentation fault|segfault|SIGSEGV)",
        severity=10,
        description="Segmentation fault",
        keywords=["segmentation fault", "segfault", "SIGSEGV"],
        suggested_action="Check for invalid memory access or null pointer dereference"
    ),
    ErrorPattern(
        pattern_id="MEMORY_004",
        category=ErrorCategory.MEMORY,
        regex=r".*(buffer overflow|stack overflow)",
        severity=9,
        description="Buffer/stack overflow",
        keywords=["overflow", "buffer", "stack"],
        suggested_action="Review buffer sizes and stack usage"
    ),
]


# Configuration Error Patterns
CONFIG_PATTERNS = [
    ErrorPattern(
        pattern_id="CONFIG_001",
        category=ErrorCategory.CONFIG,
        regex=r".*(invalid parameter|parameter.*invalid|bad parameter)",
        severity=5,
        description="Invalid parameter",
        keywords=["invalid", "parameter", "bad"],
        suggested_action="Review configuration parameters"
    ),
    ErrorPattern(
        pattern_id="CONFIG_002",
        category=ErrorCategory.CONFIG,
        regex=r".*(file not found|no such file|cannot find file).*\.cfg",
        severity=6,
        description="Configuration file not found",
        keywords=["file not found", "missing", "cfg"],
        suggested_action="Verify configuration file path and existence"
    ),
    ErrorPattern(
        pattern_id="CONFIG_003",
        category=ErrorCategory.CONFIG,
        regex=r".*(parse error|parsing failed|syntax error).*config",
        severity=6,
        description="Configuration parse error",
        keywords=["parse", "syntax", "config"],
        suggested_action="Check configuration file syntax"
    ),
    ErrorPattern(
        pattern_id="CONFIG_004",
        category=ErrorCategory.CONFIG,
        regex=r".*(missing required|required.*not found|mandatory.*missing)",
        severity=7,
        description="Missing required configuration",
        keywords=["missing", "required", "mandatory"],
        suggested_action="Add missing required configuration parameters"
    ),
]


# Assertion Error Patterns
ASSERTION_PATTERNS = [
    ErrorPattern(
        pattern_id="ASSERTION_001",
        category=ErrorCategory.ASSERTION,
        regex=r".*assertion.*failed.*at.*line\s+(\d+)",
        severity=8,
        description="Assertion failure",
        keywords=["assertion", "failed", "assert"],
        suggested_action="Review assertion condition and fix the failing logic"
    ),
    ErrorPattern(
        pattern_id="ASSERTION_002",
        category=ErrorCategory.ASSERTION,
        regex=r".*UVM_ERROR.*assertion",
        severity=8,
        description="UVM assertion error",
        keywords=["UVM_ERROR", "assertion"],
        suggested_action="Check UVM assertion and testbench logic"
    ),
    ErrorPattern(
        pattern_id="ASSERTION_003",
        category=ErrorCategory.ASSERTION,
        regex=r".*fatal.*assertion",
        severity=10,
        description="Fatal assertion",
        keywords=["fatal", "assertion"],
        suggested_action="Critical assertion failure - immediate attention required"
    ),
]


# Protocol Error Patterns
PROTOCOL_PATTERNS = [
    ErrorPattern(
        pattern_id="PROTOCOL_001",
        category=ErrorCategory.PROTOCOL,
        regex=r".*(protocol violation|protocol error)",
        severity=7,
        description="Protocol violation",
        keywords=["protocol", "violation"],
        suggested_action="Review protocol implementation and spec compliance"
    ),
    ErrorPattern(
        pattern_id="PROTOCOL_002",
        category=ErrorCategory.PROTOCOL,
        regex=r".*(handshake.*failed|handshake.*timeout)",
        severity=7,
        description="Handshake failure",
        keywords=["handshake", "failed"],
        suggested_action="Check handshake signals and timing"
    ),
    ErrorPattern(
        pattern_id="PROTOCOL_003",
        category=ErrorCategory.PROTOCOL,
        regex=r".*(invalid.*transaction|transaction.*invalid)",
        severity=6,
        description="Invalid transaction",
        keywords=["invalid", "transaction"],
        suggested_action="Verify transaction parameters and protocol rules"
    ),
]


# Compilation Error Patterns
COMPILATION_PATTERNS = [
    ErrorPattern(
        pattern_id="COMPILATION_001",
        category=ErrorCategory.COMPILATION,
        regex=r".*(syntax error|parse error).*line\s+(\d+)",
        severity=5,
        description="Syntax error",
        keywords=["syntax", "parse", "error"],
        suggested_action="Fix syntax error in source code"
    ),
    ErrorPattern(
        pattern_id="COMPILATION_002",
        category=ErrorCategory.COMPILATION,
        regex=r".*(undefined.*identifier|identifier.*not.*found)",
        severity=5,
        description="Undefined identifier",
        keywords=["undefined", "identifier"],
        suggested_action="Check variable/function definitions and imports"
    ),
    ErrorPattern(
        pattern_id="COMPILATION_003",
        category=ErrorCategory.COMPILATION,
        regex=r".*(elaboration.*failed|elab.*error)",
        severity=7,
        description="Elaboration failure",
        keywords=["elaboration", "elab", "failed"],
        suggested_action="Review design hierarchy and module instantiation"
    ),
]


# Runtime Error Patterns
RUNTIME_PATTERNS = [
    ErrorPattern(
        pattern_id="RUNTIME_001",
        category=ErrorCategory.RUNTIME,
        regex=r".*(fatal.*error|fatal)(?!.*assertion)",
        severity=9,
        description="Fatal runtime error",
        keywords=["fatal", "error"],
        suggested_action="Investigate fatal error cause and fix"
    ),
    ErrorPattern(
        pattern_id="RUNTIME_002",
        category=ErrorCategory.RUNTIME,
        regex=r".*(exception|unhandled.*exception)",
        severity=8,
        description="Unhandled exception",
        keywords=["exception", "unhandled"],
        suggested_action="Add exception handling or fix exception cause"
    ),
    ErrorPattern(
        pattern_id="RUNTIME_003",
        category=ErrorCategory.RUNTIME,
        regex=r".*(simulation.*stopped|simulation.*aborted)",
        severity=8,
        description="Simulation stopped/aborted",
        keywords=["simulation", "stopped", "aborted"],
        suggested_action="Review simulation logs for root cause"
    ),
]


# Combine all patterns
ALL_PATTERNS: List[ErrorPattern] = (
    TIMEOUT_PATTERNS +
    MEMORY_PATTERNS +
    CONFIG_PATTERNS +
    ASSERTION_PATTERNS +
    PROTOCOL_PATTERNS +
    COMPILATION_PATTERNS +
    RUNTIME_PATTERNS
)


# Pattern mapping by category
PATTERNS_BY_CATEGORY: Dict[ErrorCategory, List[ErrorPattern]] = {
    ErrorCategory.TIMEOUT: TIMEOUT_PATTERNS,
    ErrorCategory.MEMORY: MEMORY_PATTERNS,
    ErrorCategory.CONFIG: CONFIG_PATTERNS,
    ErrorCategory.ASSERTION: ASSERTION_PATTERNS,
    ErrorCategory.PROTOCOL: PROTOCOL_PATTERNS,
    ErrorCategory.COMPILATION: COMPILATION_PATTERNS,
    ErrorCategory.RUNTIME: RUNTIME_PATTERNS,
}


def match_error_pattern(error_message: str) -> Optional[Tuple[ErrorPattern, re.Match]]:
    """
    Match error message against known patterns.

    에러 메시지를 알려진 패턴과 매칭합니다.

    Args:
        error_message: Error message to match

    Returns:
        Tuple of (ErrorPattern, Match object) if matched, None otherwise
    """
    error_message_lower = error_message.lower()

    for pattern in ALL_PATTERNS:
        match = re.search(pattern.regex, error_message_lower, re.IGNORECASE)
        if match:
            return (pattern, match)

    return None


def match_error_patterns_all(error_message: str) -> List[Tuple[ErrorPattern, re.Match]]:
    """
    Match error message against all patterns.

    에러 메시지를 모든 패턴과 매칭하여 모든 매칭 결과를 반환합니다.

    Args:
        error_message: Error message to match

    Returns:
        List of (ErrorPattern, Match object) tuples
    """
    error_message_lower = error_message.lower()
    matches = []

    for pattern in ALL_PATTERNS:
        match = re.search(pattern.regex, error_message_lower, re.IGNORECASE)
        if match:
            matches.append((pattern, match))

    return matches


def classify_error_by_keywords(error_message: str) -> Optional[ErrorCategory]:
    """
    Classify error by keyword matching (fallback method).

    키워드 매칭으로 에러를 분류합니다 (패턴 매칭 실패 시 사용).

    Args:
        error_message: Error message to classify

    Returns:
        ErrorCategory if classified, None otherwise
    """
    error_message_lower = error_message.lower()

    # Count keyword matches for each category
    category_scores: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}

    for category, patterns in PATTERNS_BY_CATEGORY.items():
        for pattern in patterns:
            for keyword in pattern.keywords:
                if keyword.lower() in error_message_lower:
                    category_scores[category] += 1

    # Return category with highest score
    max_score = max(category_scores.values())
    if max_score > 0:
        for category, score in category_scores.items():
            if score == max_score:
                return category

    return None


def get_pattern_by_id(pattern_id: str) -> Optional[ErrorPattern]:
    """
    Get error pattern by ID.

    패턴 ID로 에러 패턴을 가져옵니다.

    Args:
        pattern_id: Pattern ID

    Returns:
        ErrorPattern if found, None otherwise
    """
    for pattern in ALL_PATTERNS:
        if pattern.pattern_id == pattern_id:
            return pattern
    return None


if __name__ == "__main__":
    """Test error pattern matching"""
    print("=== Testing Error Pattern Matching ===\n")

    test_messages = [
        "ERROR: Simulation timeout after 3600 seconds",
        "FATAL: Out of memory - cannot allocate 4GB",
        "ERROR: Configuration file not found: /etc/sim.cfg",
        "UVM_ERROR: Assertion failed at line 145 in testbench.sv",
        "ERROR: Protocol violation - invalid handshake sequence",
        "ERROR: Syntax error at line 89 - unexpected token",
        "FATAL: Segmentation fault at 0x7fff1234",
    ]

    for msg in test_messages:
        print(f"Message: {msg}")
        result = match_error_pattern(msg)

        if result:
            pattern, match = result
            print(f"  ✓ Matched Pattern: {pattern.pattern_id}")
            print(f"  Category: {pattern.category.value}")
            print(f"  Severity: {pattern.severity}/10")
            print(f"  Description: {pattern.description}")
            print(f"  Suggested Action: {pattern.suggested_action}")
        else:
            # Try keyword-based classification
            category = classify_error_by_keywords(msg)
            if category:
                print(f"  ~ Classified by keywords: {category.value}")
            else:
                print(f"  ✗ No pattern matched")

        print()

    print(f"\nTotal patterns defined: {len(ALL_PATTERNS)}")
    for category, patterns in PATTERNS_BY_CATEGORY.items():
        print(f"  {category.value}: {len(patterns)} patterns")
