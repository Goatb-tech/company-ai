from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[["ToolContext", dict[str, Any]], Any]


@dataclass(slots=True)
class ToolContext:
    runtime: Any
    agent_name: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Callable[
        [Callable[[ToolContext, dict[str, Any]], Any]],
        Callable[[ToolContext, dict[str, Any]], Any],
    ]:
        def decorator(
            handler: Callable[[ToolContext, dict[str, Any]], Any],
        ) -> Callable[[ToolContext, dict[str, Any]], Any]:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler,
            )
            return handler

        return decorator

    def execute(
        self, name: str, context: ToolContext, arguments: dict[str, Any]
    ) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(context, arguments)

    def describe(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        keys = names or list(self._tools.keys())
        descriptions = []
        for key in keys:
            tool = self._tools[key]
            descriptions.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        return descriptions

    def names(self) -> list[str]:
        return sorted(self._tools.keys())
