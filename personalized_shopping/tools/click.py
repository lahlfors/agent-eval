# ... license headers ...

from google.adk.tools import ToolContext
from google.genai import types

# Absolute import
from personalized_shopping.shared_libraries.init_env import webshop_env

async def click(button_name: str, tool_context: ToolContext) -> str:
    """Simulates clicking a button in the web environment."""
    # ... (function body as provided) ...
    status = {"reward": None, "done": False}
    action_string = f"click[{button_name}]"
    _, status["reward"], status["done"], _ = webshop_env.step(action_string)

    ob = webshop_env.observation
    index = ob.find("Back to Search")
    if index >= 0:
        ob = ob[index:]

    print("#" * 50)
    print("Click result:")
    print(f"status: {status}")
    # print(f"observation: {ob}")
    print("#" * 50)

    if button_name == "Back to Search":
        webshop_env.server.assigned_instruction_text = "Back to Search"

    try:
        await tool_context.save_artifact(
            "html",
            types.Part.from_uri(
                file_uri=webshop_env.state["html"], mime_type="text/html"
            ),
        )
    except ValueError as e:
        print(f"Error saving artifact: {e}")
    return ob
