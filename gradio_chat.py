import gradio as gr
from openai import OpenAI
import os
import json
from novita_sandbox.code_interpreter import Sandbox

# Create client
client = OpenAI(
    base_url="https://api.novita.ai/openai",
    api_key=os.environ["NOVITA_API_KEY"],
)

model = "meta-llama/llama-3.3-70b-instruct"

# Initialize sandbox with working directory
sandbox = Sandbox.create(timeout=1200)

# Define tool functions (no JSON serialization now)
def read_file(path: str):
    print(f"[DEBUG] read_file called with path: {path}")
    try:
        content = sandbox.files.read(path)
        print(f"[DEBUG] read_file result: {content}")
        return content
    except Exception as e:
        print(f"[DEBUG] read_file error: {e}")
        return f"Error reading file: {e}"

def write_file(path: str, data: str):
    print(f"[DEBUG] write_file called with path: {path}")
    try:
        sandbox.files.write(path, data)
        msg = f"File created successfully at {path}"
        print(f"[DEBUG] {msg}")
        return msg
    except Exception as e:
        print(f"[DEBUG] write_file error: {e}")
        return f"Error writing file: {e}"

def write_files(files: list):
    print(f"[DEBUG] write_files called with {len(files)} files")
    try:
        sandbox.files.write_files(files)
        msg = f"{len(files)} file(s) created successfully"
        print(f"[DEBUG] {msg}")
        return msg
    except Exception as e:
        print(f"[DEBUG] write_files error: {e}")
        return f"Error writing multiple files: {e}"

def run_commands(command: str):
    print(f"[DEBUG] run_commands called with command: {command}")
    try:
        result = sandbox.commands.run(command)
        print(f"[DEBUG] run_commands result: {result}")
        return result.stdout
    except Exception as e:
        print(f"[DEBUG] run_commands error: {e}")
        return f"Error running command: {e}"

# Register tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file inside the sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path in the sandbox"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a single file inside the sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path in the sandbox"},
                    "data": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_files",
            "description": "Write multiple files inside the sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "data": {"type": "string"},
                            },
                            "required": ["path", "data"],
                        },
                    }
                },
                "required": ["files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_commands",
            "description": "Run a single shell command inside the sandbox working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run, e.g. 'ls' or 'python main.py'",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

# Persistent messages
messages = []

def chat_fn(user_message, history):
    global messages
    messages.append({"role": "user", "content": user_message})

    # Send to model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    assistant_msg = response.choices[0].message
    messages.append(assistant_msg)

    output_text = ""

    if assistant_msg.tool_calls:
        print(f"[DEBUG] Assistant requested {len(assistant_msg.tool_calls)} tool call(s).")

        for tool_call in assistant_msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f"[DEBUG] Tool call detected: {fn_name} with args {fn_args}")

            if fn_name == "read_file":
                fn_result = read_file(**fn_args)
            elif fn_name == "write_file":
                fn_result = write_file(**fn_args)
            elif fn_name == "write_files":
                fn_result = write_files(**fn_args)
            elif fn_name == "run_commands":
                fn_result = run_commands(**fn_args)
            else:
                fn_result = f"Error: Unknown tool {fn_name}"
                print(f"[DEBUG] Unknown tool requested: {fn_name}")

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": str(fn_result),
            })

        follow_up = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        final_answer = follow_up.choices[0].message
        messages.append(final_answer)
        output_text = final_answer.content
    else:
        output_text = assistant_msg.content

    return output_text

with gr.Blocks() as demo:
    gr.ChatInterface(chat_fn, title="Sandbox Chat Agent")

# Graceful cleanup when the server stops
import atexit
atexit.register(lambda: (sandbox.kill(), print("[DEBUG] Sandbox terminated. 👋")))

if __name__ == "__main__":
    demo.launch()
