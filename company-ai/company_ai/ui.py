from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

from .runtime import CompanyRuntime
from .types import ModelEndpoint, RuntimeSettings
from .utils import normalize_provider_type

load_dotenv()


def build_settings(
    budget_limit_usd: float,
    ceo_provider: str,
    ceo_model: str,
    ceo_temperature: float,
    ceo_input_cost: float,
    ceo_output_cost: float,
    worker_provider: str,
    worker_model: str,
    worker_temperature: float,
    worker_input_cost: float,
    worker_output_cost: float,
) -> RuntimeSettings:
    ceo_endpoint = ModelEndpoint(
        provider=ceo_provider,
        model=ceo_model,
        temperature=ceo_temperature,
        input_cost_per_million=ceo_input_cost,
        output_cost_per_million=ceo_output_cost,
        provider_type=normalize_provider_type(ceo_provider),
    )
    worker_endpoint = ModelEndpoint(
        provider=worker_provider,
        model=worker_model,
        temperature=worker_temperature,
        input_cost_per_million=worker_input_cost,
        output_cost_per_million=worker_output_cost,
        provider_type=normalize_provider_type(worker_provider),
    )
    return RuntimeSettings(
        budget_limit_usd=budget_limit_usd,
        ceo_endpoint=ceo_endpoint,
        worker_endpoint=worker_endpoint,
    )


def create_demo() -> gr.Blocks:
    runtime = CompanyRuntime(
        build_settings(
            budget_limit_usd=10.0,
            ceo_provider="ollama",
            ceo_model="llama3.2",
            ceo_temperature=0.2,
            ceo_input_cost=0.0,
            ceo_output_cost=0.0,
            worker_provider="ollama",
            worker_model="llama3.2",
            worker_temperature=0.2,
            worker_input_cost=0.0,
            worker_output_cost=0.0,
        )
    )

    def apply_runtime_settings(
        *values: float | str,
    ) -> tuple[str, str, str, list[list[str]]]:
        runtime.update_settings(build_settings(*values))
        return (
            runtime.settings_markdown(),
            runtime.team_markdown(),
            runtime.budget_markdown(),
            runtime.activity_rows(),
        )

    def reset_runtime(
        *values: float | str,
    ) -> tuple[list[dict[str, str]], str, str, str, list[list[str]]]:
        runtime.reset(build_settings(*values))
        return (
            [],
            runtime.settings_markdown(),
            runtime.team_markdown(),
            runtime.budget_markdown(),
            runtime.activity_rows(),
        )

    def send_message(message: str, history: list[dict[str, str]], *values: float | str):
        history = history or []
        if not message.strip():
            return (
                history,
                "",
                runtime.settings_markdown(),
                runtime.team_markdown(),
                runtime.budget_markdown(),
                runtime.activity_rows(),
            )
        runtime.update_settings(build_settings(*values))
        response = runtime.chat(message)
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return (
            updated_history,
            "",
            runtime.settings_markdown(),
            runtime.team_markdown(),
            runtime.budget_markdown(),
            runtime.activity_rows(),
        )

    with gr.Blocks(title="Company AI") as demo:
        gr.Markdown(
            "# Company AI\n"
            "Chat with a CEO agent that can spin up employee agents, delegate tasks, track budget, and retire workers when the job is done."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=580, label="CEO Console")
                message_box = gr.Textbox(
                    label="Message the CEO",
                    placeholder="Ask the CEO to plan, delegate, or build something.",
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    reset_button = gr.Button("Reset Company")

            with gr.Column(scale=2):
                gr.Markdown("## Runtime Settings")
                budget_limit = gr.Number(
                    label="Overall Budget (USD)", value=10.0, minimum=0
                )
                with gr.Row():
                    ceo_provider = gr.Dropdown(
                        choices=["ollama", "groq", "gemini"],
                        value="ollama",
                        label="CEO Provider",
                    )
                    ceo_model = gr.Textbox(label="CEO Model", value="llama3.2")
                with gr.Row():
                    ceo_temperature = gr.Slider(
                        label="CEO Temperature",
                        minimum=0,
                        maximum=1,
                        value=0.2,
                        step=0.05,
                    )
                    ceo_input_cost = gr.Number(
                        label="CEO Input $/1M", value=0.0, minimum=0
                    )
                    ceo_output_cost = gr.Number(
                        label="CEO Output $/1M", value=0.0, minimum=0
                    )
                with gr.Row():
                    worker_provider = gr.Dropdown(
                        choices=["ollama", "groq", "gemini"],
                        value="ollama",
                        label="Worker Provider",
                    )
                    worker_model = gr.Textbox(label="Worker Model", value="llama3.2")
                with gr.Row():
                    worker_temperature = gr.Slider(
                        label="Worker Temperature",
                        minimum=0,
                        maximum=1,
                        value=0.2,
                        step=0.05,
                    )
                    worker_input_cost = gr.Number(
                        label="Worker Input $/1M", value=0.0, minimum=0
                    )
                    worker_output_cost = gr.Number(
                        label="Worker Output $/1M", value=0.0, minimum=0
                    )
                apply_button = gr.Button("Apply Settings")
                settings_markdown = gr.Markdown(runtime.settings_markdown())
                team_markdown = gr.Markdown(runtime.team_markdown(), label="Team")
                budget_markdown = gr.Markdown(runtime.budget_markdown(), label="Budget")
                activity_table = gr.Dataframe(
                    headers=["Time", "Agent", "Type", "Message"],
                    datatype=["str", "str", "str", "str"],
                    value=runtime.activity_rows(),
                    interactive=False,
                    label="Activity Feed",
                    wrap=True,
                )

        controls = [
            budget_limit,
            ceo_provider,
            ceo_model,
            ceo_temperature,
            ceo_input_cost,
            ceo_output_cost,
            worker_provider,
            worker_model,
            worker_temperature,
            worker_input_cost,
            worker_output_cost,
        ]

        apply_button.click(
            apply_runtime_settings,
            inputs=controls,
            outputs=[settings_markdown, team_markdown, budget_markdown, activity_table],
        )
        send_button.click(
            send_message,
            inputs=[message_box, chatbot, *controls],
            outputs=[
                chatbot,
                message_box,
                settings_markdown,
                team_markdown,
                budget_markdown,
                activity_table,
            ],
        )
        message_box.submit(
            send_message,
            inputs=[message_box, chatbot, *controls],
            outputs=[
                chatbot,
                message_box,
                settings_markdown,
                team_markdown,
                budget_markdown,
                activity_table,
            ],
        )
        reset_button.click(
            reset_runtime,
            inputs=controls,
            outputs=[
                chatbot,
                settings_markdown,
                team_markdown,
                budget_markdown,
                activity_table,
            ],
        )

    return demo
