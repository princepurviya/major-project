import sys
sys.stdout.reconfigure(encoding="utf-8")

# Windows fix: must be BEFORE any asyncio import
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import asyncio
import argparse
import json
import os
import time
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Support both SDKs
try:
    from google import genai
    from google.genai import types
    _SDK_MODE = "google-genai"
except ImportError:
    _SDK_MODE = "none"

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def log(msg: str):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


AGENT_SYSTEM_PROMPT = """
You are an autonomous EDA agent. When given a CSV file path, you:
1. First use get_quick_stats to understand what you're working with
2. Then use run_full_eda_pipeline to generate the complete report
3. Confirm to the user what was done and where the report was saved

Be decisive — don't ask for confirmation, just run the analysis.
If a tool fails, explain the error clearly and suggest a fix.
"""


def mcp_to_gemini_tools(mcp_tools):
    """Convert MCP tool definitions to Gemini function declarations."""
    gemini_func_declarations = []
    for t in mcp_tools:
        schema = t.get("input_schema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        clean_props = {}
        for p_name, p_val in properties.items():
            prop_type = p_val.get("type", "STRING").upper()
            clean_props[p_name] = {
                "type": prop_type,
                "description": p_val.get("description", "")
            }

        func_decl = {
            "name": t["name"],
            "description": t["description"].strip(),
            "parameters": {
                "type": "OBJECT",
                "properties": clean_props,
                "required": required
            }
        }
        gemini_func_declarations.append(func_decl)

    return [{"function_declarations": gemini_func_declarations}]


async def run_agent(csv_filepath: str):
    if _SDK_MODE != "google-genai":
        log("ERROR: google-genai SDK is required for MCP client. pip install google-genai")
        return

    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=os.environ.copy()
    )

    log(f"\n🚀 Starting EDA Agent (Gemini {GEMINI_MODEL})")
    log(f"📂 Target file: {csv_filepath}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            log("✅ MCP server connected\n")

            tools_result = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema if hasattr(t, "inputSchema") else (t.input_schema if hasattr(t, "input_schema") else {})
                }
                for t in tools_result.tools
            ]
            log(f"🔧 Tools available: {[t['name'] for t in tools]}\n")

            gemini_tools = mcp_to_gemini_tools(tools)

            # Create Gemini client and chat — single model, no fallback loop
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                log("ERROR: GEMINI_API_KEY not set in .env")
                return

            client = genai.Client(api_key=api_key)

            try:
                chat = client.chats.create(
                    model=GEMINI_MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction=AGENT_SYSTEM_PROMPT,
                        tools=gemini_tools,
                        temperature=0.2,
                    )
                )
            except Exception as e:
                log(f"ERROR: Failed to create Gemini chat with model '{GEMINI_MODEL}': {e}")
                return

            prompt = f"Please run a complete EDA analysis on this file: {csv_filepath}"
            turn = 0

            log(f"📤 Sending initial prompt to Gemini...")
            start_time = time.time()
            response = chat.send_message(prompt)
            log(f"   Response received in {time.time() - start_time:.1f}s")

            while True:
                turn += 1
                log(f"--- Agent turn {turn} ---")

                # Check if model wants to call function(s)
                function_calls = response.function_calls

                if function_calls:
                    function_responses = []
                    for call in function_calls:
                        name = call.name
                        args = dict(call.args) if call.args else {}
                        log(f"\n🔧 Gemini calling: {name}")
                        log(f"   With args: {json.dumps(args, indent=2)}")

                        tool_start = time.time()
                        result = await session.call_tool(name, args)
                        result_text = result.content[0].text
                        log(f"   Tool completed in {time.time() - tool_start:.1f}s")
                        log(f"   Result preview: {result_text[:150]}...")

                        function_responses.append(
                            types.Part.from_function_response(
                                name=name,
                                response={"result": result_text}
                            )
                        )

                    log(f"📤 Sending tool results to Gemini...")
                    start_time = time.time()
                    response = chat.send_message(function_responses)
                    log(f"   Response received in {time.time() - start_time:.1f}s")
                else:
                    log("\n✅ Agent finished!\n")
                    if response.text:
                        print(response.text)
                    break

                if turn > 10:
                    log("⚠️ Max turns reached, stopping.")
                    break


def main():
    parser = argparse.ArgumentParser(description="EDA Agent — Automated Data Analysis")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_agent(args.file))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
