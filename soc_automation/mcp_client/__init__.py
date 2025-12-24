"""
MCP Client Loader for SOC Automation System.

MCP 서버 연결 및 도구 로드
- stdio 방식으로 MCP 서버 연결
- LangChain tools로 변환
- 여러 MCP 서버 동시 관리
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.langchain import load_mcp_tools

from ..utils.logger import get_logger


logger = get_logger()


class MCPClientLoader:
    """
    MCP Client Loader.

    여러 MCP 서버를 연결하고 도구를 로드합니다.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize MCP Client Loader.

        Args:
            config_path: Path to MCP servers configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "mcp_servers.json"

        self.config_path = Path(config_path)
        self.stack = None
        self.sessions: List[ClientSession] = []
        self.all_tools = []

    async def __aenter__(self):
        """Async context manager entry."""
        await self.load_all_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def load_config(self) -> Dict[str, Any]:
        """
        Load MCP servers configuration.

        MCP 서버 설정을 로드합니다.

        Returns:
            Dict: MCP servers configuration
        """
        if not self.config_path.exists():
            logger.warning(f"MCP config not found: {self.config_path}")
            return {"servers": []}

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded MCP config from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {"servers": []}

    async def load_all_servers(self) -> List[Any]:
        """
        Load all MCP servers and their tools.

        모든 MCP 서버를 로드하고 도구를 가져옵니다.

        Returns:
            List: All loaded tools
        """
        config = self.load_config()
        servers = config.get("servers", [])

        if not servers:
            logger.warning("No MCP servers configured")
            return []

        self.stack = AsyncExitStack()
        await self.stack.__aenter__()

        for server_config in servers:
            try:
                await self._load_server(server_config)
            except Exception as e:
                logger.error(f"Failed to load MCP server {server_config.get('name')}: {e}")
                continue

        logger.info(f"Loaded {len(self.all_tools)} tools from {len(self.sessions)} MCP servers")
        return self.all_tools

    async def _load_server(self, server_config: Dict[str, Any]) -> None:
        """
        Load a single MCP server.

        단일 MCP 서버를 로드합니다.

        Args:
            server_config: Server configuration
        """
        name = server_config.get("name", "unknown")
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        if not command:
            logger.error(f"Server {name}: missing 'command'")
            return

        logger.info(f"Loading MCP server: {name}")
        logger.debug(f"  Command: {command} {' '.join(args)}")

        try:
            # Create StdioServerParameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env if env else None
            )

            # Connect to server via stdio
            read, write = await self.stack.enter_async_context(
                stdio_client(server_params)
            )

            # Create client session
            session = await self.stack.enter_async_context(
                ClientSession(read, write)
            )

            # Initialize session
            await session.initialize()

            # Load tools from this server
            tools = await load_mcp_tools(session)

            self.sessions.append(session)
            self.all_tools.extend(tools)

            logger.info(f"  ✓ Loaded {len(tools)} tools from {name}")

            # Log tool names
            for tool in tools:
                logger.debug(f"    - {tool.name}")

        except Exception as e:
            logger.error(f"  ✗ Failed to load {name}: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """
        Close all MCP connections.

        모든 MCP 연결을 종료합니다.
        """
        if self.stack:
            try:
                await self.stack.__aexit__(None, None, None)
                logger.info("Closed all MCP server connections")
            except Exception as e:
                logger.error(f"Error closing MCP connections: {e}")

    def get_tools_by_server(self, server_name: str) -> List[Any]:
        """
        Get tools from a specific server.

        특정 서버의 도구만 가져옵니다.

        Args:
            server_name: Name of the server

        Returns:
            List: Tools from the specified server
        """
        # Note: This is a simple implementation
        # In practice, you'd need to track which tools came from which server
        return self.all_tools

    def get_all_tools(self) -> List[Any]:
        """
        Get all loaded tools.

        모든 로드된 도구를 가져옵니다.

        Returns:
            List: All tools
        """
        return self.all_tools


async def load_mcp_tools_async(config_path: Optional[Path] = None) -> Tuple[AsyncExitStack, List[Any]]:
    """
    Load MCP tools (standalone function).

    MCP 도구를 로드합니다 (독립 함수).

    Args:
        config_path: Path to MCP servers configuration file

    Returns:
        Tuple[AsyncExitStack, List]: Stack and loaded tools

    Example:
        ```python
        stack, tools = await load_mcp_tools_async()
        # Use tools...
        await stack.aclose()
        ```
    """
    loader = MCPClientLoader(config_path)
    await loader.load_all_servers()
    return loader.stack, loader.all_tools


if __name__ == "__main__":
    """Test MCP client loader"""
    import sys

    async def test():
        print("=== MCP Client Loader Test ===\n")

        # Load MCP servers
        async with MCPClientLoader() as loader:
            tools = loader.get_all_tools()

            print(f"Loaded {len(tools)} tools from MCP servers\n")

            if tools:
                print("Available tools:")
                for i, tool in enumerate(tools, 1):
                    print(f"{i}. {tool.name}")
                    if hasattr(tool, 'description'):
                        print(f"   Description: {tool.description}")
                    print()
            else:
                print("No tools loaded. Check your MCP server configuration.")
                print(f"Config file: {loader.config_path}")
                print("\nMake sure:")
                print("1. MCP servers are properly configured in mcp_servers.json")
                print("2. MCP server commands are correct and executable")
                print("3. All required dependencies are installed")

    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
