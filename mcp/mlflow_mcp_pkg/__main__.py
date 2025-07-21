"""Entry point for MLflow MCP server."""

import asyncio
import logging
import sys

from .server import MLflowMCPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)


async def async_main():
    """Run the MLflow MCP server."""
    server = MLflowMCPServer()
    await server.run()


def main():
    """Entry point for console script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
