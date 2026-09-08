from __future__ import annotations

from .registry import ToolContext, ToolRegistry


def register_custom_tools(registry: ToolRegistry) -> None:
    @registry.register(
        name="agent_note",
        description="Store a note in the shared activity feed.",
        input_schema={
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Note to persist in the activity log",
                }
            },
            "required": ["note"],
        },
    )
    def agent_note(context: ToolContext, arguments: dict) -> dict:
        note = arguments["note"]
        context.runtime.log(context.agent_name, "custom_tool_note", note)
        return {"saved": True, "note": note}
