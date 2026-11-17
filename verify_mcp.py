import sys
sys.stdout.reconfigure(encoding="utf-8")

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_client_server():
    print("--- Testing Real MCP Client <-> MCP Server over stdio ---")
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=os.environ.copy()
    )

    csv_path = os.path.join("data", "test_sales.csv")
    if not os.path.exists(csv_path):
        import pandas as pd
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({"product": ["A", "B"], "price": [10, 20]}).to_csv(csv_path, index=False)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize MCP handshake
            print("1. Sending initialize request to MCP Server...")
            init_res = await session.initialize()
            print("   ✅ MCP Server Handshake Successful!")
            print(f"   Server Name: {init_res.serverInfo.name if hasattr(init_res, 'serverInfo') else 'eda-agent'}")

            # 2. List tools from MCP Server
            print("\n2. Discovering tools via session.list_tools()...")
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"   ✅ Discovered {len(tool_names)} tools on MCP Server:")
            for name in tool_names:
                print(f"      - {name}")

            # 3. Call execute_sql_query via JSON-RPC
            print("\n3. Calling 'execute_sql_query' tool via MCP JSON-RPC protocol...")
            call_res = await session.call_tool("execute_sql_query", {
                "filepath": csv_path,
                "query": "SELECT COUNT(*) as total_records FROM dataset"
            })
            print(f"   ✅ Tool Call Returned from MCP Server:")
            print(f"   {call_res.content[0].text.strip()}")

            # 4. Call analyze_data_quality via JSON-RPC
            print("\n4. Calling 'analyze_data_quality' tool via MCP JSON-RPC protocol...")
            dq_res = await session.call_tool("analyze_data_quality", {"filepath": csv_path})
            print(f"   ✅ Data Quality Tool Call Returned from MCP Server:")
            parsed = json.loads(dq_res.content[0].text)
            print(f"   Score: {parsed['quality_score']['overall_score']} / 100")

    print("\n🎉 MCP Server & Client stdio Protocol Verification: 100% OPERATIONAL!")

if __name__ == "__main__":
    asyncio.run(test_mcp_client_server())
