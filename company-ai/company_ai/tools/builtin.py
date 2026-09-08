from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from company_ai.utils import safe_calculate

from .custom import register_custom_tools
from .registry import ToolContext, ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        name="get_time",
        description="Return the current timestamp in a timezone.",
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone like UTC or Asia/Kolkata",
                }
            },
        },
    )
    def get_time(_: ToolContext, arguments: dict) -> dict:
        timezone_name = arguments.get("timezone", "UTC")
        current = datetime.now(ZoneInfo(timezone_name))
        return {"timezone": timezone_name, "iso_time": current.isoformat()}

    @registry.register(
        name="calculator",
        description="Evaluate a basic arithmetic expression.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression"}
            },
            "required": ["expression"],
        },
    )
    def calculator(_: ToolContext, arguments: dict) -> dict:
        expression = arguments["expression"]
        return {"expression": expression, "result": safe_calculate(expression)}

    @registry.register(
        name="budget_snapshot",
        description="Inspect current budget status and spend records.",
        input_schema={"type": "object", "properties": {}},
    )
    def budget_snapshot(context: ToolContext, _: dict) -> dict:
        return context.runtime.budget.summary()

    @registry.register(
        name="list_team",
        description="List active employee agents and their roles.",
        input_schema={"type": "object", "properties": {}},
    )
    def list_team(context: ToolContext, _: dict) -> dict:
        return context.runtime.team_snapshot()

    register_custom_tools(registry)

    return registry
