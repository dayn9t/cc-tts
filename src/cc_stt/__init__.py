import asyncio
from .server import main

def cli():
    """CLI entry point"""
    asyncio.run(main())

__all__ = ["cli", "main"]
