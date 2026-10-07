from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

client = MultiServerMCPClient(
    {
        "cortex":{
            "transport":"stdio",
            "command":"python",
            "args":["mcp_server.py"],
        }
    }
)


async def main():
    tools = await client.get_tools()
    for tool in tools:
        print(tool)
        
asyncio.run(main())