from __future__ import annotations

import json
from copy import deepcopy

from .budget import BudgetManager
from .providers import ProviderFactory
from .tools import ToolContext, build_default_registry
from .types import (
    ActivityEvent,
    AgentProfile,
    ModelEndpoint,
    RuntimeSettings,
    ToolCall,
    utc_now,
)
from .utils import extract_json_object, normalize_provider_type, render_markdown_list


class CompanyRuntime:
    SUPPORTED_PROVIDERS = {"ollama", "groq", "gemini"}

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.budget = BudgetManager(settings.budget_limit_usd)
        self.tool_registry = build_default_registry()
        self.ceo_history: list[dict[str, str]] = []
        self.activities: list[ActivityEvent] = []
        self.employees: dict[str, AgentProfile] = {}
        self._employee_counter = 0

    def reset(self, settings: RuntimeSettings | None = None) -> None:
        if settings is not None:
            self.settings = settings
        self.budget = BudgetManager(self.settings.budget_limit_usd)
        self.ceo_history = []
        self.activities = []
        self.employees = {}
        self._employee_counter = 0

    def update_settings(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.budget.set_limit(settings.budget_limit_usd)

    def log(self, agent: str, kind: str, message: str, **metadata: object) -> None:
        self.activities.append(
            ActivityEvent(
                timestamp=utc_now(),
                agent=agent,
                kind=kind,
                message=message,
                metadata=dict(metadata),
            )
        )

    def team_snapshot(self) -> dict:
        return {
            "employee_count": len(self.employees),
            "employees": [
                {
                    "name": employee.name,
                    "role": employee.role,
                    "status": employee.status,
                    "provider": employee.endpoint.provider,
                    "model": employee.endpoint.model,
                    "last_result": employee.last_result,
                }
                for employee in self.employees.values()
            ],
        }

    def activity_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for event in self.activities[-50:]:
            rows.append(
                [
                    event.timestamp.strftime("%H:%M:%S"),
                    event.agent,
                    event.kind,
                    event.message,
                ]
            )
        return rows

    def budget_markdown(self) -> str:
        summary = self.budget.summary()
        return "\n".join(
            [
                f"**Budget Limit:** `${summary['limit_usd']:.4f}`",
                f"**Spent:** `${summary['spent_usd']:.4f}`",
                f"**Remaining:** `${summary['remaining_usd']:.4f}`",
                f"**Cloud Spend:** `${summary['cloud_spend_usd']:.4f}`",
                f"**Local Calls:** `{summary['local_calls']}`",
            ]
        )

    def team_markdown(self) -> str:
        if not self.employees:
            return "No active employees."
        blocks = []
        for employee in self.employees.values():
            blocks.append(
                "\n".join(
                    [
                        f"**{employee.name}**",
                        f"Role: `{employee.role}`",
                        f"Status: `{employee.status}`",
                        f"Model: `{employee.endpoint.provider}:{employee.endpoint.model}`",
                        f"Last result: {employee.last_result or 'None'}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def chat(self, user_message: str) -> str:
        self.log("CEO", "user_request", user_message)
        self.ceo_history.append({"role": "user", "content": user_message})
        final_response = self._run_ceo_loop(user_message)
        self.ceo_history.append({"role": "assistant", "content": final_response})
        self.log("CEO", "final_response", final_response)
        return final_response

    def _call_model(
        self, agent_name: str, endpoint: ModelEndpoint, messages: list[dict[str, str]]
    ) -> str:
        if not self.budget.can_spend(endpoint):
            message = f"Budget exhausted for cloud provider {endpoint.provider}:{endpoint.model}."
            self.log(agent_name, "budget_block", message)
            return message
        try:
            provider = ProviderFactory.get(endpoint.provider)
            response = provider.generate(endpoint, messages)
            self.budget.record_usage(agent_name, endpoint, response.usage)
            self.log(
                agent_name,
                "model_call",
                f"Used {endpoint.provider}:{endpoint.model}",
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            return response.text
        except Exception as exc:
            message = (
                f"Model call failed for {endpoint.provider}:{endpoint.model}: {exc}"
            )
            self.log(agent_name, "model_error", message)
            return message

    def _build_ceo_prompt(self) -> str:
        budget_summary = self.budget.summary()
        team = self.team_snapshot()
        return "\n".join(
            [
                "You are the CEO of a small AI company.",
                "You manage employee agents to complete the user's request.",
                "Create employees only when delegation helps, prefer short-lived specialists, and delete them when done.",
                "Prefer local models when budget is tight.",
                f"Current budget summary: {json.dumps(budget_summary)}",
                f"Current team: {json.dumps(team)}",
                f"Available employee tools: {json.dumps(self.tool_registry.describe())}",
                "Return JSON only with this shape:",
                json.dumps(
                    {
                        "analysis": "brief string",
                        "actions": [
                            {
                                "type": "create_employee",
                                "name": "Researcher",
                                "role": "API specialist",
                                "provider": "optional provider override",
                                "model": "optional model override",
                                "input_cost_per_million": 0.0,
                                "output_cost_per_million": 0.0,
                            },
                            {
                                "type": "assign_task",
                                "employee": "Researcher",
                                "task": "Do the work",
                            },
                            {"type": "delete_employee", "employee": "Researcher"},
                            {
                                "type": "final_response",
                                "content": "Message for the user",
                            },
                        ],
                    }
                ),
                "Use at most 5 actions per turn.",
                "Valid providers are only: ollama, groq, gemini.",
                "If you are unsure about provider or model overrides, reuse the default worker settings.",
                "Always include a final_response once the work is complete.",
            ]
        )

    def _run_ceo_loop(self, user_message: str) -> str:
        planning_history = deepcopy(self.ceo_history)
        final_response = ""

        for _ in range(4):
            messages = [
                {"role": "system", "content": self._build_ceo_prompt()}
            ] + planning_history
            raw_plan = self._call_model("CEO", self.settings.ceo_endpoint, messages)
            try:
                plan = extract_json_object(raw_plan)
            except Exception:
                return raw_plan

            self.log("CEO", "plan", plan.get("analysis", "Planning step"))
            actions = plan.get("actions", [])
            action_notes: list[str] = []

            for action in actions[:5]:
                action_type = action.get("type", "")
                if action_type == "create_employee":
                    action_notes.append(self._create_employee(action))
                elif action_type == "assign_task":
                    action_notes.append(self._assign_task(action))
                elif action_type == "delete_employee":
                    action_notes.append(self._delete_employee(action))
                elif action_type == "final_response":
                    final_response = action.get("content", "")

            if final_response:
                return final_response

            planning_history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(plan),
                }
            )
            planning_history.append(
                {
                    "role": "user",
                    "content": "Execution results:\n" + "\n".join(action_notes),
                }
            )

        return self._synthesize_fallback_response(user_message, planning_history)

    def _synthesize_fallback_response(
        self, user_message: str, planning_history: list[dict[str, str]]
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are the CEO. The planning loop finished without a final_response. Reply directly to the user with a concise, useful answer based on the execution trace. Do not return JSON.",
            },
            {"role": "user", "content": f"Original request: {user_message}"},
            {
                "role": "user",
                "content": "Execution trace:\n" + json.dumps(planning_history[-8:]),
            },
        ]
        response = self._call_model("CEO", self.settings.ceo_endpoint, messages)
        if response and not response.startswith("Model call failed"):
            return response
        return "The CEO finished the management loop but could not produce a final answer. Check the activity feed for the execution trace."

    def _resolve_worker_endpoint(
        self, action: dict
    ) -> tuple[ModelEndpoint, str | None]:
        base_endpoint = self.settings.worker_endpoint
        requested_provider = str(action.get("provider", base_endpoint.provider)).lower()
        provider_note = None

        if requested_provider not in self.SUPPORTED_PROVIDERS:
            provider_note = f"Unsupported provider override '{requested_provider}' ignored; used {base_endpoint.provider}."
            requested_provider = base_endpoint.provider

        return (
            ModelEndpoint(
                provider=requested_provider,
                model=action.get("model", base_endpoint.model),
                temperature=base_endpoint.temperature,
                max_output_tokens=base_endpoint.max_output_tokens,
                input_cost_per_million=float(
                    action.get(
                        "input_cost_per_million", base_endpoint.input_cost_per_million
                    )
                ),
                output_cost_per_million=float(
                    action.get(
                        "output_cost_per_million", base_endpoint.output_cost_per_million
                    )
                ),
                provider_type=normalize_provider_type(requested_provider),
            ),
            provider_note,
        )

    def _create_employee(self, action: dict) -> str:
        name = action.get("name") or f"Employee-{self._employee_counter + 1}"
        if name in self.employees:
            return f"Employee {name} already exists."

        self._employee_counter += 1
        endpoint, provider_note = self._resolve_worker_endpoint(action)
        employee = AgentProfile(
            agent_id=f"employee-{self._employee_counter}",
            name=name,
            role=action.get("role", "General specialist"),
            endpoint=endpoint,
            tools=self.tool_registry.names(),
        )
        self.employees[name] = employee
        self.log("CEO", "create_employee", f"Created {name}", role=employee.role)
        if provider_note:
            self.log("CEO", "provider_override_ignored", provider_note)
            return f"Created employee {name} for role {employee.role}. {provider_note}"
        return f"Created employee {name} for role {employee.role}."

    def _assign_task(self, action: dict) -> str:
        employee_name = action.get("employee", "")
        task = action.get("task", "")
        employee = self.employees.get(employee_name)
        if employee is None:
            return f"Employee {employee_name} does not exist."
        result = self._run_employee_task(employee, task)
        return f"{employee_name} completed task. Result: {result}"

    def _delete_employee(self, action: dict) -> str:
        employee_name = action.get("employee", "")
        if employee_name not in self.employees:
            return f"Employee {employee_name} does not exist."
        del self.employees[employee_name]
        self.log("CEO", "delete_employee", f"Deleted {employee_name}")
        return f"Deleted employee {employee_name}."

    def _run_employee_task(self, employee: AgentProfile, task: str) -> str:
        employee.status = "working"
        self.log(employee.name, "task_started", task)

        planner_prompt = "\n".join(
            [
                f"You are {employee.name}, role: {employee.role}.",
                "Solve the assigned task directly when possible.",
                "If you need tools, request them explicitly.",
                f"Available tools: {json.dumps(self.tool_registry.describe(employee.tools))}",
                "Return JSON only with this shape:",
                json.dumps(
                    {
                        "thought": "brief string",
                        "tool_calls": [
                            {"tool": "calculator", "arguments": {"expression": "2+2"}}
                        ],
                        "final_response": "leave empty if waiting for tools",
                    }
                ),
            ]
        )
        initial_messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": task},
        ]
        raw_plan = self._call_model(employee.name, employee.endpoint, initial_messages)

        try:
            plan = extract_json_object(raw_plan)
        except Exception:
            employee.status = "idle"
            employee.last_result = raw_plan
            self.log(employee.name, "task_completed", raw_plan)
            return raw_plan

        tool_calls = [
            ToolCall(tool=item.get("tool", ""), arguments=item.get("arguments", {}))
            for item in plan.get("tool_calls", [])
        ]
        final_response = plan.get("final_response", "")

        if not tool_calls and not final_response:
            final_response = plan.get("thought", "") or raw_plan

        if tool_calls:
            tool_outputs = []
            for tool_call in tool_calls[:3]:
                try:
                    result = self.tool_registry.execute(
                        tool_call.tool,
                        ToolContext(runtime=self, agent_name=employee.name),
                        tool_call.arguments,
                    )
                    self.log(employee.name, "tool_call", f"Used tool {tool_call.tool}")
                    tool_outputs.append({"tool": tool_call.tool, "result": result})
                except Exception as exc:
                    error = {"tool": tool_call.tool, "error": str(exc)}
                    self.log(
                        employee.name, "tool_error", f"Tool {tool_call.tool} failed"
                    )
                    tool_outputs.append(error)

            followup_messages = [
                {
                    "role": "system",
                    "content": f"You are {employee.name}, role: {employee.role}. Produce the final answer for the task.",
                },
                {"role": "user", "content": f"Task: {task}"},
                {
                    "role": "user",
                    "content": f"Tool results: {json.dumps(tool_outputs)}",
                },
            ]
            final_response = self._call_model(
                employee.name, employee.endpoint, followup_messages
            )

        employee.status = "idle"
        employee.last_result = final_response
        self.log(employee.name, "task_completed", final_response)
        return final_response

    def settings_markdown(self) -> str:
        return "\n".join(
            [
                f"**CEO Model:** `{self.settings.ceo_endpoint.provider}:{self.settings.ceo_endpoint.model}`",
                f"**Worker Model:** `{self.settings.worker_endpoint.provider}:{self.settings.worker_endpoint.model}`",
                f"**Worker Tools:**\n{render_markdown_list(self.tool_registry.names())}",
            ]
        )
