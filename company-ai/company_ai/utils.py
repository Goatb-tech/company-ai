from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from typing import Any


def normalize_provider_type(provider: str) -> str:
    return "local" if provider.lower() == "ollama" else "cloud"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        stripped = next((part for part in parts if "{" in part), stripped)
        stripped = stripped.replace("json", "", 1).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    for index in range(start, len(stripped)):
        char = stripped[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("Parsed JSON is not an object")

    raise ValueError("Incomplete JSON object in model output")


def render_markdown_list(items: Iterable[str]) -> str:
    values = [item for item in items if item]
    if not values:
        return "- None"
    return "\n".join(f"- {item}" for item in values)


def safe_calculate(expression: str) -> float:
    node = ast.parse(expression, mode="eval")

    def evaluate(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return evaluate(current.body)
        if isinstance(current, ast.Constant) and isinstance(
            current.value, (int, float)
        ):
            return float(current.value)
        if isinstance(current, ast.UnaryOp) and isinstance(
            current.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(current.operand)
            return value if isinstance(current.op, ast.UAdd) else -value
        if isinstance(current, ast.BinOp) and isinstance(
            current.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            left = evaluate(current.left)
            right = evaluate(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            if isinstance(current.op, ast.Div):
                return left / right
            if isinstance(current.op, ast.FloorDiv):
                return left // right
            if isinstance(current.op, ast.Mod):
                return left % right
            return left**right
        raise ValueError("Unsupported expression")

    return evaluate(node)
