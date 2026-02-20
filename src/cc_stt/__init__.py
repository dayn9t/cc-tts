import asyncio

def cli():
    """CLI entry point for MCP server"""
    from .server import main
    asyncio.run(main())

def daemon_cli():
    """CLI entry point for voice assistant daemon"""
    from .daemon import main as daemon_main
    daemon_main()

__all__ = ["cli", "daemon_cli"]
