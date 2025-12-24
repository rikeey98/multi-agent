"""
Test MCP client loader.

MCP 클라이언트 로더 테스트
"""

import asyncio
import sys

from . import MCPClientLoader


async def test():
    """Test MCP client loader"""
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


def main():
    """Main entry point"""
    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
