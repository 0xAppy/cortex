from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio


async def main():
    # Define how we launch the "mcp_server.py"
    cortex = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    # Start a stdio client session that communicates with our MCP server!
    async with stdio_client(cortex) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session 
            await session.initialize()
            print("Connected to the MCP Server!")
            
            tools = await session.list_tools()
            print(tools)
            
            response_tool = await session.call_tool(
                name = "retrieve_documents",
                arguments={"question": "Why does Cortex use ChromaDB?"}
            )
            
            result = response_tool.content[0].text
            
            print(result)
            
            
        
if __name__ == "__main__": 
    asyncio.run(main())