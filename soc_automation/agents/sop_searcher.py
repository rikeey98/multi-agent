"""
SOP Searcher Agent for SOC automation system.

SOP 검색 Agent
- SOP 문서 검색
- 해결 절차 추출
- 자동화 가능성 판단
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..config.agent_prompts import get_agent_prompt
from ..utils.state import SOPResult
from ..utils.logger import get_agent_logger


# Tools for SOP Searcher Agent
@tool
def search_sop_directory(sop_dir: str, keywords: List[str]) -> str:
    """
    Search SOP documents in directory by keywords.

    SOP 디렉토리에서 키워드로 문서를 검색합니다.

    Args:
        sop_dir: Path to SOP directory
        keywords: List of keywords to search

    Returns:
        str: Found SOP documents and their relevance
    """
    try:
        sop_path = Path(sop_dir)
        if not sop_path.exists():
            return f"SOP directory not found: {sop_dir}"

        # Search for .md, .txt files
        sop_files = list(sop_path.glob("**/*.md")) + list(sop_path.glob("**/*.txt"))

        if not sop_files:
            return f"No SOP documents found in {sop_dir}"

        results = []
        for sop_file in sop_files:
            try:
                with open(sop_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()

                # Calculate relevance score
                matches = sum(1 for kw in keywords if kw.lower() in content)
                if matches > 0:
                    relevance = matches / len(keywords)
                    results.append({
                        "file": str(sop_file),
                        "name": sop_file.name,
                        "relevance": relevance,
                        "matches": matches
                    })
            except Exception as e:
                continue

        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)

        if not results:
            return f"No SOP documents matched the keywords: {keywords}"

        # Format results
        output = f"Found {len(results)} relevant SOP documents:\n\n"
        for i, result in enumerate(results[:10], 1):
            output += f"{i}. {result['name']} (relevance: {result['relevance']:.2f}, matches: {result['matches']})\n"
            output += f"   Path: {result['file']}\n"

        return output

    except Exception as e:
        return f"Error searching SOP directory: {str(e)}"


@tool
def read_sop_document(sop_file_path: str) -> str:
    """
    Read SOP document content.

    SOP 문서 내용을 읽습니다.

    Args:
        sop_file_path: Path to SOP document

    Returns:
        str: SOP document content
    """
    try:
        path = Path(sop_file_path)
        if not path.exists():
            return f"SOP document not found: {sop_file_path}"

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return f"SOP Document: {path.name}\n{'='*80}\n{content}"

    except Exception as e:
        return f"Error reading SOP document: {str(e)}"


@tool
def extract_resolution_steps(sop_content: str) -> str:
    """
    Extract resolution steps from SOP document.

    SOP 문서에서 해결 단계를 추출합니다.

    Args:
        sop_content: SOP document content

    Returns:
        str: Extracted resolution steps
    """
    try:
        # Look for numbered steps or bullet points
        lines = sop_content.split('\n')
        steps = []

        # Pattern for numbered steps
        numbered_pattern = re.compile(r'^\s*(\d+[\.\)])\s+(.+)')
        # Pattern for bullet points
        bullet_pattern = re.compile(r'^\s*[-*]\s+(.+)')

        in_steps_section = False
        for line in lines:
            # Check if we're in a steps/resolution/procedure section
            if re.search(r'(steps|resolution|procedure|solution|fix)', line.lower()):
                in_steps_section = True

            if in_steps_section:
                numbered_match = numbered_pattern.match(line)
                bullet_match = bullet_pattern.match(line)

                if numbered_match:
                    steps.append(f"Step {numbered_match.group(1)} {numbered_match.group(2)}")
                elif bullet_match:
                    steps.append(f"- {bullet_match.group(1)}")

        if not steps:
            return "No explicit resolution steps found. Manual review required."

        return "Extracted Resolution Steps:\n" + '\n'.join(steps)

    except Exception as e:
        return f"Error extracting steps: {str(e)}"


@tool
def assess_automation_feasibility(resolution_steps: str) -> str:
    """
    Assess if resolution steps can be automated.

    해결 단계의 자동화 가능성을 평가합니다.

    Args:
        resolution_steps: Resolution steps text

    Returns:
        str: Automation feasibility assessment
    """
    try:
        content_lower = resolution_steps.lower()

        # Safe operations (automatable)
        safe_keywords = [
            "restart", "rerun", "update config", "change parameter",
            "set variable", "clear cache", "delete temp", "cleanup"
        ]

        # Risky operations (manual only)
        risky_keywords = [
            "delete database", "drop table", "modify schema",
            "change network", "update firewall", "manual review",
            "verify manually", "contact", "escalate"
        ]

        safe_count = sum(1 for kw in safe_keywords if kw in content_lower)
        risky_count = sum(1 for kw in risky_keywords if kw in content_lower)

        if risky_count > 0:
            feasibility = "LOW"
            reason = f"Contains risky operations requiring manual intervention (found {risky_count} risky keywords)"
        elif safe_count > 0:
            feasibility = "HIGH"
            reason = f"Contains automatable operations (found {safe_count} safe keywords)"
        else:
            feasibility = "MEDIUM"
            reason = "No explicit automation indicators found - requires review"

        return f"""
Automation Feasibility Assessment:
- Feasibility: {feasibility}
- Reason: {reason}
- Safe operations detected: {safe_count}
- Risky operations detected: {risky_count}
"""

    except Exception as e:
        return f"Error assessing feasibility: {str(e)}"


# Default tools for SOP searcher
DEFAULT_TOOLS = [
    search_sop_directory,
    read_sop_document,
    extract_resolution_steps,
    assess_automation_feasibility,
]


def create_sop_searcher_agent(llm, tools: Optional[List] = None):
    """
    Create SOP Searcher Agent.

    SOP 검색 Agent를 생성합니다.

    Args:
        llm: Language model to use
        tools: List of tools (default: DEFAULT_TOOLS)

    Returns:
        Agent: SOP Searcher Agent
    """
    if tools is None:
        tools = DEFAULT_TOOLS

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=get_agent_prompt("sop_searcher")
    )

    return agent


async def run_sop_searcher(
    llm,
    error_type: str,
    error_message: str,
    sop_dir: str,
    tools: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run SOP searcher agent.

    SOP 검색 Agent를 실행합니다.

    Args:
        llm: Language model
        error_type: Type of error (e.g., "TIMEOUT")
        error_message: Error message
        sop_dir: SOP documents directory
        tools: Optional tools list

    Returns:
        Dict: SOP search results
    """
    logger = get_agent_logger("sop_searcher")
    logger.info(f"Searching SOP for error type: {error_type}")

    try:
        # Create agent
        agent = create_sop_searcher_agent(llm, tools)

        # Prepare input message
        input_message = f"""
Please search for SOP documents related to this error:

Error Type: {error_type}
Error Message: {error_message}
SOP Directory: {sop_dir}

Tasks:
1. Search SOP directory for relevant documents using keywords from error type and message
2. Read the most relevant SOP documents
3. Extract resolution steps from the documents
4. Assess automation feasibility for the resolution steps
5. Identify any warnings or prerequisites

Provide a structured result with:
- sop_id: Document identifier
- title: SOP document title
- relevance_score: 0-1 relevance score
- resolution_steps: List of steps to resolve the issue
- automation_feasible: Whether auto-execution is possible (true/false)
- warnings: Any warnings or cautions from the SOP
- prerequisites: Required conditions or tools
"""

        # Run agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]}
        )

        logger.info("SOP search completed")

        # Extract results from agent output
        messages = result.get("messages", [])
        final_message = messages[-1].content if messages else "No SOP found"

        return {
            "status": "success",
            "error_type": error_type,
            "sop_results": final_message,
            "raw_result": result
        }

    except Exception as e:
        logger.error(f"SOP search failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": error_type,
            "error": str(e)
        }


if __name__ == "__main__":
    """
    Test SOP Searcher Agent independently.

    독립적으로 SOP Searcher Agent를 테스트합니다.
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    print("=== SOP Searcher Agent Test ===\n")

    # Create sample SOP directory and documents
    sop_dir = Path("/tmp/test_sop")
    sop_dir.mkdir(exist_ok=True)

    # Sample SOP for timeout errors
    timeout_sop = """# SOP-001: Simulation Timeout Resolution

## Problem
Simulation timeout errors occur when the simulation exceeds the allocated time limit.

## Resolution Steps
1. Check for infinite loops in the testbench or DUT
2. Review clock gating logic
3. Increase simulation timeout in configuration file
4. Update timeout parameter in sim.cfg: timeout = 7200
5. Rerun the simulation
6. Verify simulation completes successfully

## Prerequisites
- Access to simulation configuration files
- Understanding of clock domains

## Warnings
- Do not set timeout too high as it may mask real issues
- Always review simulation logs after timeout changes

## Automation
Steps 3-5 can be automated safely.
"""

    sop_file = sop_dir / "timeout_resolution.md"
    with open(sop_file, 'w') as f:
        f.write(timeout_sop)

    print(f"Created sample SOP: {sop_file}\n")

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

    # Run SOP searcher
    async def test():
        result = await run_sop_searcher(
            llm=llm,
            error_type="TIMEOUT",
            error_message="Simulation timeout after 3600 seconds",
            sop_dir=str(sop_dir)
        )

        print("\n=== SOP Search Result ===\n")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"\nSOP Results:\n{result['sop_results']}")
        else:
            print(f"Error: {result.get('error')}")

    # Run test
    asyncio.run(test())

    print(f"\nTest completed. SOP directory: {sop_dir}")
