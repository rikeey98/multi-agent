"""
MCP Client Helper Functions.

MCP 클라이언트 헬퍼 함수들
- 설정 파일 생성
- MCP 서버 검증
- 도구 필터링
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils.logger import get_logger


logger = get_logger()


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand environment variables in configuration.

    설정의 환경 변수를 확장합니다.

    Args:
        config: Configuration dictionary

    Returns:
        Dict: Configuration with expanded environment variables
    """
    if "env" in config and isinstance(config["env"], dict):
        expanded_env = {}
        for key, value in config["env"].items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                expanded_env[key] = os.getenv(env_var, value)
            else:
                expanded_env[key] = value
        config["env"] = expanded_env

    return config


def filter_enabled_servers(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter enabled servers from configuration.

    활성화된 서버만 필터링합니다.

    Args:
        config: MCP servers configuration

    Returns:
        List: Enabled servers
    """
    servers = config.get("servers", [])
    enabled_servers = [
        expand_env_vars(server)
        for server in servers
        if server.get("enabled", True)  # Default to enabled if not specified
    ]

    logger.info(f"Found {len(enabled_servers)} enabled MCP servers out of {len(servers)} total")
    return enabled_servers


def create_sample_config(output_path: Path) -> None:
    """
    Create a sample MCP servers configuration file.

    샘플 MCP 서버 설정 파일을 생성합니다.

    Args:
        output_path: Path to save the configuration file
    """
    sample_config = {
        "servers": [
            {
                "name": "my_mongodb",
                "description": "My MongoDB MCP Server",
                "command": "python",
                "args": ["-m", "my_mcp_servers.mongodb"],
                "env": {
                    "MONGODB_URI": "${MONGODB_URI}",
                    "MONGODB_DB": "${MONGODB_DB}"
                },
                "enabled": True
            },
            {
                "name": "my_oracle",
                "description": "My Oracle MCP Server",
                "command": "python",
                "args": ["-m", "my_mcp_servers.oracle"],
                "env": {
                    "ORACLE_USER": "${ORACLE_USER}",
                    "ORACLE_PASSWORD": "${ORACLE_PASSWORD}",
                    "ORACLE_DSN": "${ORACLE_DSN}"
                },
                "enabled": True
            }
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    logger.info(f"Created sample MCP config: {output_path}")


def validate_server_config(server_config: Dict[str, Any]) -> bool:
    """
    Validate MCP server configuration.

    MCP 서버 설정을 검증합니다.

    Args:
        server_config: Server configuration

    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ["name", "command"]

    for field in required_fields:
        if field not in server_config:
            logger.error(f"Server config missing required field: {field}")
            return False

    return True


def filter_tools_by_prefix(tools: List[Any], prefix: str) -> List[Any]:
    """
    Filter tools by name prefix.

    이름 접두사로 도구를 필터링합니다.

    Args:
        tools: List of tools
        prefix: Prefix to filter by

    Returns:
        List: Filtered tools
    """
    filtered = [tool for tool in tools if tool.name.startswith(prefix)]
    logger.debug(f"Filtered {len(filtered)} tools with prefix '{prefix}'")
    return filtered


def get_tool_names(tools: List[Any]) -> List[str]:
    """
    Get list of tool names.

    도구 이름 목록을 가져옵니다.

    Args:
        tools: List of tools

    Returns:
        List[str]: Tool names
    """
    return [tool.name for tool in tools]


def print_tools_summary(tools: List[Any]) -> None:
    """
    Print summary of loaded tools.

    로드된 도구 요약을 출력합니다.

    Args:
        tools: List of tools
    """
    print(f"\n{'='*80}")
    print(f"Loaded {len(tools)} MCP Tools")
    print(f"{'='*80}\n")

    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}")
        if hasattr(tool, 'description') and tool.description:
            # Truncate long descriptions
            desc = tool.description
            if len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"   {desc}")
        print()


if __name__ == "__main__":
    """Test helper functions"""
    print("=== MCP Client Helpers Test ===\n")

    # Test environment variable expansion
    test_config = {
        "name": "test",
        "command": "python",
        "env": {
            "DB_URI": "${MONGODB_URI}",
            "API_KEY": "literal_value"
        }
    }

    print("Original config:")
    print(json.dumps(test_config, indent=2))

    expanded = expand_env_vars(test_config)
    print("\nExpanded config:")
    print(json.dumps(expanded, indent=2))

    # Test validation
    print("\n--- Validation Test ---")
    valid_config = {"name": "test", "command": "python"}
    invalid_config = {"command": "python"}  # Missing name

    print(f"Valid config: {validate_server_config(valid_config)}")
    print(f"Invalid config: {validate_server_config(invalid_config)}")

    # Test sample config creation
    print("\n--- Sample Config Creation ---")
    sample_path = Path("/tmp/mcp_servers_sample.json")
    create_sample_config(sample_path)
    print(f"Sample config created at: {sample_path}")

    if sample_path.exists():
        with open(sample_path) as f:
            sample = json.load(f)
        print("\nSample config content:")
        print(json.dumps(sample, indent=2))
