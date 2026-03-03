from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import gradio as gr

try:
    from .agent import GradioChatAgent
except ImportError:
    from agent import GradioChatAgent


@dataclass
class SessionConfig:
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"


CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_SAVED_AGENT = {
    "name": "OpenAI Default",
    "agent": {
        "implementation": "openai",
        "model": "gpt-4o-mini",
        "base_url": None,
        "api_key": None,
        "api_key_env": "OPENAI_API_KEY",
    },
}


def create_app(
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
    api_key_env: str = "OPENAI_API_KEY",
) -> gr.Blocks:
    return _build_chat_app(
        SessionConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env=api_key_env,
        ),
        show_start_screen=False,
    )


def create_start_app() -> gr.Blocks:
    saved_agents, _ = load_saved_agents()
    default_config = session_config_from_saved_agent(saved_agents[0])
    return _build_chat_app(default_config, show_start_screen=True)


def launch_app(
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    **kwargs,
) -> gr.Blocks:
    app = create_app(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    app.launch(**kwargs)
    return app


def launch_start_app(**kwargs) -> gr.Blocks:
    app = create_start_app()
    app.launch(**kwargs)
    return app


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _build_chat_app(default_config: SessionConfig, show_start_screen: bool) -> gr.Blocks:
    saved_agents, saved_agents_notice = load_saved_agents()
    saved_agent_names = [agent["name"] for agent in saved_agents]
    default_saved_agent_name = saved_agent_names[0] if saved_agent_names else None

    with gr.Blocks(title="Gradio Chatbot Demo") as app:
        session_config_state = gr.State(asdict(default_config))
        agent_state = gr.State(None)
        chat_history_state = gr.State([])
        debug_log_state = gr.State([])
        model_context_state = gr.State([])
        session_closed_state = gr.State(False)
        internals_visible_state = gr.State(False)

        gr.Markdown("# Gradio LangGraph Chatbot Demo")

        with gr.Column(visible=show_start_screen) as start_screen:
            gr.Markdown("## Start Session")
            with gr.Row():
                with gr.Column(scale=2):
                    start_model = gr.Textbox(label="Model", value=default_config.model)
                    start_api_key = gr.Textbox(
                        label="API Key",
                        value=default_config.api_key,
                        type="password",
                    )
                    start_api_key_env = gr.Textbox(
                        label="API Key Env",
                        value=default_config.api_key_env,
                    )
                    start_base_url = gr.Textbox(
                        label="Base URL",
                        value=default_config.base_url,
                    )
                    start_button = gr.Button("Start Chat", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Saved Agents")
                    saved_agents_info = gr.Markdown(saved_agents_notice)
                    saved_agents_radio = gr.Radio(
                        label="Saved Agent",
                        choices=saved_agent_names,
                        value=default_saved_agent_name,
                    )

        with gr.Column(visible=not show_start_screen) as chat_screen:
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=520,
                    )
                    with gr.Row():
                        message_box = gr.Textbox(
                            label="Message",
                            placeholder="Type a message...",
                            lines=3,
                        )
                    with gr.Row():
                        send_button = gr.Button(
                            "Send",
                            variant="primary",
                            interactive=False,
                        )
                        quit_button = gr.Button("Quit", variant="stop")
                        internals_button = gr.Button("Internals")

                with gr.Column(scale=2, visible=False) as internals_panel:
                    debug_output = gr.Textbox(
                        label="Verbose / State Flow",
                        lines=16,
                        max_lines=24,
                        interactive=False,
                    )
                    context_output = gr.Textbox(
                        label="Exact Model Context",
                        lines=16,
                        max_lines=24,
                        interactive=False,
                    )

            status_output = gr.Textbox(
                label="Status",
                value="Ready.",
                interactive=False,
            )

        def update_send_button(message: str, session_closed: bool) -> gr.Button:
            interactive = bool((message or "").strip()) and not session_closed
            return gr.update(interactive=interactive)

        def populate_saved_agent(agent_name: str):
            agent_config = find_saved_agent(saved_agents, agent_name)
            if agent_config is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            return (
                gr.update(value=agent_config.model),
                gr.update(value=agent_config.api_key),
                gr.update(value=agent_config.api_key_env),
                gr.update(value=agent_config.base_url),
            )

        def start_session(
            model: str,
            api_key: str,
            api_key_env: str,
            base_url: str,
        ):
            config = {
                "model": model.strip() or default_config.model,
                "api_key": api_key,
                "api_key_env": api_key_env.strip(),
                "base_url": base_url.strip(),
            }
            return (
                config,
                None,
                [],
                [],
                [],
                False,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=[]),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value="Ready."),
                gr.update(value=""),
                gr.update(interactive=False),
            )

        def toggle_internals(visible: bool):
            next_visible = not visible
            return next_visible, gr.update(visible=next_visible)

        def submit_message(
            user_input: str,
            current_agent_state,
            chat_history: list[dict[str, str]],
            debug_log: list[str],
            model_context: list[dict[str, str]],
            session_config: dict[str, str],
            session_closed: bool,
        ):
            if session_closed:
                return (
                    current_agent_state,
                    chat_history,
                    debug_log,
                    model_context,
                    True,
                    gr.update(value=chat_history),
                    _format_debug_log(debug_log),
                    _format_model_context(model_context),
                    "Session closed. Reload or use the start screen to begin a new chat.",
                    gr.update(value=""),
                    gr.update(interactive=False),
                )

            agent = GradioChatAgent(
                model=session_config["model"],
                api_key=session_config["api_key"] or None,
                base_url=session_config["base_url"] or None,
                api_key_env=session_config.get("api_key_env") or None,
            )
            initial_state = current_agent_state or agent.get_initial_state()
            result = agent.run_turn(initial_state, user_input)

            next_history = result["full_history"]
            next_debug_log = debug_log + result["debug_log"]
            next_model_context = result["model_context"]

            if result["session_closed"] and (
                not next_history or next_history[-1]["role"] != "assistant"
            ):
                next_history = next_history + [
                    {
                        "role": "assistant",
                        "content": "Session closed.",
                    }
                ]

            status = "Session closed." if result["session_closed"] else "Ready."
            return (
                result["state"],
                next_history,
                next_debug_log,
                next_model_context,
                result["session_closed"],
                gr.update(value=next_history),
                _format_debug_log(next_debug_log),
                _format_model_context(next_model_context),
                status,
                gr.update(value=""),
                gr.update(interactive=False),
            )

        def quit_session(
            current_agent_state,
            chat_history: list[dict[str, str]],
            debug_log: list[str],
            model_context: list[dict[str, str]],
            session_config: dict[str, str],
            session_closed: bool,
        ):
            if session_closed:
                return (
                    current_agent_state,
                    chat_history,
                    debug_log,
                    model_context,
                    True,
                    gr.update(value=chat_history),
                    _format_debug_log(debug_log),
                    _format_model_context(model_context),
                    "Session closed.",
                    gr.update(value=""),
                    gr.update(interactive=False),
                )

            return submit_message(
                "quit",
                current_agent_state,
                chat_history,
                debug_log,
                model_context,
                session_config,
                session_closed,
            )

        if show_start_screen:
            saved_agents_radio.change(
                fn=populate_saved_agent,
                inputs=[saved_agents_radio],
                outputs=[
                    start_model,
                    start_api_key,
                    start_api_key_env,
                    start_base_url,
                ],
            )

            start_button.click(
                fn=start_session,
                inputs=[
                    start_model,
                    start_api_key,
                    start_api_key_env,
                    start_base_url,
                ],
                outputs=[
                    session_config_state,
                    agent_state,
                    chat_history_state,
                    debug_log_state,
                    model_context_state,
                    session_closed_state,
                    start_screen,
                    internals_panel,
                    chat_screen,
                    chatbot,
                    debug_output,
                    context_output,
                    status_output,
                    message_box,
                    send_button,
                ],
            )

        message_box.change(
            fn=update_send_button,
            inputs=[message_box, session_closed_state],
            outputs=send_button,
        )

        send_button.click(
            fn=submit_message,
            inputs=[
                message_box,
                agent_state,
                chat_history_state,
                debug_log_state,
                model_context_state,
                session_config_state,
                session_closed_state,
            ],
            outputs=[
                agent_state,
                chat_history_state,
                debug_log_state,
                model_context_state,
                session_closed_state,
                chatbot,
                debug_output,
                context_output,
                status_output,
                message_box,
                send_button,
            ],
        )

        quit_button.click(
            fn=quit_session,
            inputs=[
                agent_state,
                chat_history_state,
                debug_log_state,
                model_context_state,
                session_config_state,
                session_closed_state,
            ],
            outputs=[
                agent_state,
                chat_history_state,
                debug_log_state,
                model_context_state,
                session_closed_state,
                chatbot,
                debug_output,
                context_output,
                status_output,
                message_box,
                send_button,
            ],
        )

        internals_button.click(
            fn=toggle_internals,
            inputs=internals_visible_state,
            outputs=[internals_visible_state, internals_panel],
        )

    return app


def _format_debug_log(debug_log: list[str]) -> str:
    if not debug_log:
        return "No verbose events yet."
    return "\n".join(debug_log)


def _format_model_context(model_context: list[dict[str, str]]) -> str:
    if not model_context:
        return json.dumps([], indent=2)
    return json.dumps(model_context, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Gradio LangGraph chatbot demo."
    )
    parser.add_argument(
        "--mode",
        choices=("start", "direct"),
        default="start",
        help="Use the start screen or launch directly into chat.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name passed to ChatOpenAI.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenAI-compatible API key for direct mode or start-screen default.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name used when --api-key is not provided.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Launch with a public Gradio share link.",
    )
    parser.add_argument(
        "--inbrowser",
        dest="inbrowser",
        action="store_true",
        help="Open the Gradio app in the browser after launch.",
    )
    parser.add_argument(
        "--no-inbrowser",
        dest="inbrowser",
        action="store_false",
        help="Do not open a browser window after launch.",
    )
    parser.set_defaults(inbrowser=None)
    parser.add_argument(
        "--server-name",
        default=None,
        help="Optional Gradio server host, for example 127.0.0.1 or 0.0.0.0.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Optional Gradio server port.",
    )
    return parser.parse_args()


def main() -> gr.Blocks:
    args = parse_args()
    launch_kwargs = {
        "share": args.share,
    }

    if args.inbrowser is None:
        launch_kwargs["inbrowser"] = not is_colab()
    else:
        launch_kwargs["inbrowser"] = args.inbrowser

    if args.server_name:
        launch_kwargs["server_name"] = args.server_name
    if args.server_port is not None:
        launch_kwargs["server_port"] = args.server_port

    if args.mode == "direct":
        return launch_app(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            **launch_kwargs,
        )

    return launch_start_app(**launch_kwargs)


def load_saved_agents() -> tuple[list[dict], str]:
    saved_agents: list[dict] = []

    if CONFIGS_DIR.exists():
        for path in sorted(CONFIGS_DIR.glob("*.json")):
            agent_config = load_saved_agent_file(path)
            if agent_config is not None:
                saved_agents.append(agent_config)

    if not saved_agents:
        saved_agents = [DEFAULT_SAVED_AGENT]
        notice = (
            "No saved agent configs were found. Using the built-in default agent."
        )
    else:
        notice = f"Loaded {len(saved_agents)} saved agent configuration(s)."

    return saved_agents, notice


def load_saved_agent_file(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    agent_payload = payload.get("agent", payload)
    if not isinstance(agent_payload, dict):
        return None

    return {
        "name": payload.get("name", path.stem),
        "agent": {
            "implementation": agent_payload.get("implementation", "openai"),
            "model": agent_payload.get("model", "gpt-4o-mini"),
            "base_url": agent_payload.get("base_url"),
            "api_key": agent_payload.get("api_key"),
            "api_key_env": agent_payload.get("api_key_env", "OPENAI_API_KEY"),
        },
    }


def session_config_from_saved_agent(saved_agent: dict) -> SessionConfig:
    agent_payload = saved_agent["agent"]
    return SessionConfig(
        model=agent_payload.get("model") or "gpt-4o-mini",
        api_key=agent_payload.get("api_key") or "",
        base_url=agent_payload.get("base_url") or "",
        api_key_env=agent_payload.get("api_key_env") or "",
    )


def find_saved_agent(saved_agents: list[dict], name: str) -> SessionConfig | None:
    for saved_agent in saved_agents:
        if saved_agent["name"] == name:
            return session_config_from_saved_agent(saved_agent)
    return None


if __name__ == "__main__":
    main()
