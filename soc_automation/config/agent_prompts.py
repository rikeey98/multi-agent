"""
Agent-specific prompts for SOC automation system.

각 Agent별 시스템 프롬프트 정의
프롬프트는 개별 MD 파일로 관리됩니다.
"""

from pathlib import Path
from typing import Dict


# Prompts directory
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(agent_name: str) -> str:
    """
    Load prompt from markdown file.

    Args:
        agent_name: Name of the agent

    Returns:
        str: Prompt content

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_file = PROMPTS_DIR / f"{agent_name}.md"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


# Available agent names
AVAILABLE_AGENTS = [
    "supervisor",
    "error_analyzer",
    "sop_searcher",
    "data_collector",
    "decision_maker",
    "auto_executor",
    "notification",
]


def get_agent_prompt(agent_name: str) -> str:
    """
    Get prompt for specific agent.

    특정 Agent의 프롬프트를 가져옵니다.

    Args:
        agent_name: Name of the agent

    Returns:
        str: Agent prompt

    Raises:
        ValueError: If agent name is not found
        FileNotFoundError: If prompt file doesn't exist
    """
    if agent_name not in AVAILABLE_AGENTS:
        raise ValueError(
            f"Unknown agent: {agent_name}. "
            f"Available agents: {AVAILABLE_AGENTS}"
        )

    return _load_prompt(agent_name)


def list_available_agents() -> list:
    """
    List all available agents.

    Returns:
        list: List of agent names
    """
    return AVAILABLE_AGENTS.copy()


if __name__ == "__main__":
    """Test agent prompts"""
    print("=== Available Agent Prompts ===\n")

    for agent_name in AVAILABLE_AGENTS:
        try:
            prompt = get_agent_prompt(agent_name)
            print(f"Agent: {agent_name}")
            print(f"Prompt file: {PROMPTS_DIR / f'{agent_name}.md'}")
            print(f"Prompt length: {len(prompt)} characters")
            print(f"First 200 chars: {prompt[:200]}...")
            print("-" * 80)
            print()
        except Exception as e:
            print(f"Error loading {agent_name}: {e}")
            print("-" * 80)
            print()
