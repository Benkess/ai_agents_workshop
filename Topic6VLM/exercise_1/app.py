from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
from typing import Any

import gradio as gr

try:
    from .agent import DEFAULT_SYSTEM_PROMPT, GradioChatAgent
except ImportError:
    from agent import DEFAULT_SYSTEM_PROMPT, GradioChatAgent


APP_TITLE = "Vision-Language LangGraph Chatbot"


@dataclass
class SessionConfig:
    model: str = "gpt-5.2"
    api_key: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    trim_strategy: str = "last"
    token_counter: str = "approximate"
    max_tokens: int = 16384
    start_on: str = "human"
    include_system: bool = True
    allow_partial: bool = False


CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_SAVED_AGENT = {
    "name": "OpenAI GPT-5.2",
    "agent": {
        "implementation": "openai",
        "model": "gpt-5.2",
        "base_url": None,
        "api_key": None,
        "api_key_env": "OPENAI_API_KEY",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "trim_strategy": "last",
        "token_counter": "approximate",
        "max_tokens": 16384,
        "start_on": "human",
        "include_system": True,
        "allow_partial": False,
    },
}


def create_app(
    model: str = "gpt-5.2",
    api_key: str = "",
    base_url: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    trim_strategy: str = "last",
    token_counter: str = "approximate",
    max_tokens: int = 16384,
    start_on: str = "human",
    include_system: bool = True,
    allow_partial: bool = False,
) -> gr.Blocks:
    return _build_chat_app(
        SessionConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env=api_key_env,
            system_prompt=system_prompt,
            trim_strategy=trim_strategy,
            token_counter=token_counter,
            max_tokens=max_tokens,
            start_on=start_on,
            include_system=include_system,
            allow_partial=allow_partial,
        ),
        show_start_screen=False,
    )


def create_start_app() -> gr.Blocks:
    saved_agents, _ = load_saved_agents()
    default_config = session_config_from_saved_agent(saved_agents[0])
    return _build_chat_app(default_config, show_start_screen=True)


def launch_app(
    model: str = "gpt-5.2",
    api_key: str = "",
    base_url: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    trim_strategy: str = "last",
    token_counter: str = "approximate",
    max_tokens: int = 16384,
    start_on: str = "human",
    include_system: bool = True,
    allow_partial: bool = False,
    **kwargs,
) -> gr.Blocks:
    app = create_app(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_key_env=api_key_env,
        system_prompt=system_prompt,
        trim_strategy=trim_strategy,
        token_counter=token_counter,
        max_tokens=max_tokens,
        start_on=start_on,
        include_system=include_system,
        allow_partial=allow_partial,
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

    with gr.Blocks(title=APP_TITLE) as app:
        session_config_state = gr.State(asdict(default_config))
        agent_state = gr.State(None)
        uploaded_image_state = gr.State("")
        chat_history_state = gr.State([])
        debug_log_state = gr.State([])
        model_context_state = gr.State([])
        session_closed_state = gr.State(False)
        internals_visible_state = gr.State(False)
        advanced_visible_state = gr.State(False)

        with gr.Column():
            gr.Markdown(
                "# Vision-Language LangGraph Chatbot\n"
                "Upload one image, then chat about it. The uploaded image stays fixed for the session."
            )

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
                        advanced_button = gr.Button("Advanced Settings")
                        with gr.Column(visible=False) as advanced_panel:
                            start_system_prompt = gr.Textbox(
                                label="System Prompt",
                                value=default_config.system_prompt,
                                lines=4,
                            )
                            with gr.Row():
                                start_trim_strategy = gr.Textbox(
                                    label="Trim Strategy",
                                    value=default_config.trim_strategy,
                                )
                                start_token_counter = gr.Textbox(
                                    label="Token Counter",
                                    value=default_config.token_counter,
                                )
                            with gr.Row():
                                start_max_tokens = gr.Number(
                                    label="Max Tokens",
                                    value=default_config.max_tokens,
                                    precision=0,
                                )
                                start_start_on = gr.Textbox(
                                    label="Start On",
                                    value=default_config.start_on,
                                )
                            with gr.Row():
                                start_include_system = gr.Checkbox(
                                    label="Include System",
                                    value=default_config.include_system,
                                )
                                start_allow_partial = gr.Checkbox(
                                    label="Allow Partial",
                                    value=default_config.allow_partial,
                                )
                        with gr.Row():
                            start_button = gr.Button(
                                "Continue to Image Upload",
                                variant="primary",
                            )
                            close_button = gr.Button("Close App", variant="stop")

                    with gr.Column(scale=1):
                        gr.Markdown("### Saved Agents")
                        saved_agents_info = gr.Markdown(saved_agents_notice)
                        saved_agents_radio = gr.Radio(
                            label="Saved Agent",
                            choices=saved_agent_names,
                            value=default_saved_agent_name,
                        )

            with gr.Column(visible=not show_start_screen) as upload_screen:
                gr.Markdown("## Session Image")
                gr.Markdown(
                    "Choose the image for this chat. To switch images later, return here and start a new session."
                )
                image_input = gr.File(
                    label="Image File",
                    type="filepath",
                    file_types=["image"],
                )
                upload_status = gr.Textbox(
                    label="Session Status",
                    value="Upload an image to continue.",
                    interactive=False,
                )
                with gr.Row():
                    upload_next_button = gr.Button(
                        "Start Chat",
                        variant="primary",
                        interactive=False,
                    )
                    upload_back_button = gr.Button("Back", visible=show_start_screen)

            with gr.Column(visible=False) as chat_screen:
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="Conversation", height=520)
                        message_box = gr.Textbox(
                            label="Message",
                            placeholder="Ask a question about the uploaded image...",
                            lines=3,
                        )
                        with gr.Row():
                            send_button = gr.Button(
                                "Send",
                                variant="primary",
                                interactive=False,
                            )
                            quit_button = gr.Button("New Session", variant="stop")
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

            with gr.Column():
                status_output = gr.Textbox(
                    label="Status",
                    value="Ready.",
                    interactive=False,
                )

        def update_send_button(message: str, session_closed: bool):
            interactive = bool((message or "").strip()) and not session_closed
            return gr.update(interactive=interactive)

        def populate_saved_agent(agent_name: str):
            agent_config = find_saved_agent(saved_agents, agent_name)
            if agent_config is None:
                return tuple(gr.update() for _ in range(11))

            return (
                gr.update(value=agent_config.model),
                gr.update(value=agent_config.api_key),
                gr.update(value=agent_config.api_key_env),
                gr.update(value=agent_config.base_url),
                gr.update(value=agent_config.system_prompt),
                gr.update(value=agent_config.trim_strategy),
                gr.update(value=agent_config.token_counter),
                gr.update(value=agent_config.max_tokens),
                gr.update(value=agent_config.start_on),
                gr.update(value=agent_config.include_system),
                gr.update(value=agent_config.allow_partial),
            )

        def toggle_advanced(visible: bool):
            next_visible = not visible
            return next_visible, gr.update(visible=next_visible)

        def toggle_internals(visible: bool):
            next_visible = not visible
            return next_visible, gr.update(visible=next_visible)

        def build_config(
            model: str,
            api_key: str,
            api_key_env: str,
            base_url: str,
            system_prompt: str,
            trim_strategy: str,
            token_counter: str,
            max_tokens: float,
            start_on: str,
            include_system: bool,
            allow_partial: bool,
        ) -> dict[str, Any]:
            return {
                "model": model.strip() or default_config.model,
                "api_key": api_key,
                "api_key_env": api_key_env.strip(),
                "base_url": base_url.strip(),
                "system_prompt": system_prompt.strip() or default_config.system_prompt,
                "trim_strategy": trim_strategy.strip() or default_config.trim_strategy,
                "token_counter": token_counter.strip() or default_config.token_counter,
                "max_tokens": int(max_tokens),
                "start_on": start_on.strip() or default_config.start_on,
                "include_system": bool(include_system),
                "allow_partial": bool(allow_partial),
            }

        def build_restored_config(session_config: dict[str, Any]) -> SessionConfig:
            return SessionConfig(
                model=session_config.get("model") or default_config.model,
                api_key=session_config.get("api_key") or "",
                base_url=session_config.get("base_url") or "",
                api_key_env=session_config.get("api_key_env") or "",
                system_prompt=(
                    session_config.get("system_prompt") or default_config.system_prompt
                ),
                trim_strategy=(
                    session_config.get("trim_strategy") or default_config.trim_strategy
                ),
                token_counter=(
                    session_config.get("token_counter") or default_config.token_counter
                ),
                max_tokens=int(session_config.get("max_tokens", default_config.max_tokens)),
                start_on=session_config.get("start_on") or default_config.start_on,
                include_system=bool(
                    session_config.get("include_system", default_config.include_system)
                ),
                allow_partial=bool(
                    session_config.get("allow_partial", default_config.allow_partial)
                ),
            )

        def reset_session_outputs(show_start: bool, advanced_visible: bool):
            base_updates: list[Any] = [
                None,
                "",
                [],
                [],
                [],
                False,
                False,
                gr.update(visible=show_start),
                gr.update(visible=advanced_visible),
                gr.update(visible=not show_start),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=None, interactive=True),
                gr.update(value="Upload an image to continue."),
                gr.update(interactive=False),
                gr.update(value=[]),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value="Ready."),
                gr.update(value=""),
                gr.update(interactive=False),
            ]
            return tuple(base_updates)

        def go_to_upload(
            model: str,
            api_key: str,
            api_key_env: str,
            base_url: str,
            system_prompt: str,
            trim_strategy: str,
            token_counter: str,
            max_tokens: float,
            start_on: str,
            include_system: bool,
            allow_partial: bool,
        ):
            config = build_config(
                model=model,
                api_key=api_key,
                api_key_env=api_key_env,
                base_url=base_url,
                system_prompt=system_prompt,
                trim_strategy=trim_strategy,
                token_counter=token_counter,
                max_tokens=max_tokens,
                start_on=start_on,
                include_system=include_system,
                allow_partial=allow_partial,
            )
            return (config,) + reset_session_outputs(show_start=False, advanced_visible=False)

        def back_to_start(advanced_visible: bool):
            return reset_session_outputs(show_start=True, advanced_visible=advanced_visible)

        def update_upload_button(image_path: str | None):
            if image_path and Path(image_path).is_file():
                filename = Path(image_path).name
                return (
                    gr.update(interactive=True),
                    gr.update(
                        value=(
                            f"Selected image: {filename}. Click Start Chat to lock this "
                            "image into the session."
                        )
                    ),
                )
            return (
                gr.update(interactive=False),
                gr.update(value="Upload an image to continue."),
            )

        def open_chat(image_path: str | None):
            if not image_path:
                return (
                    "",
                    False,
                    gr.update(visible=False),
                    gr.update(value="Please upload an image before continuing."),
                    gr.update(),
                    gr.update(interactive=False),
                )
            path = Path(image_path)
            if not path.is_file():
                return (
                    "",
                    False,
                    gr.update(visible=False),
                    gr.update(value=f"Image file not found: {image_path}"),
                    gr.update(),
                    gr.update(interactive=False),
                )
            return (
                image_path,
                False,
                gr.update(visible=True),
                gr.update(
                    value=(
                        f"Session locked to {path.name}. Use New Session to choose another image."
                    )
                ),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        def create_agent(session_config: dict[str, Any]) -> GradioChatAgent:
            return GradioChatAgent(
                model=session_config["model"],
                api_key=session_config["api_key"] or None,
                base_url=session_config["base_url"] or None,
                api_key_env=session_config.get("api_key_env") or None,
                system_prompt=session_config.get("system_prompt"),
                trim_strategy=session_config.get("trim_strategy"),
                token_counter=session_config.get("token_counter"),
                max_tokens=session_config.get("max_tokens"),
                start_on=session_config.get("start_on"),
                include_system=session_config.get("include_system"),
                allow_partial=session_config.get("allow_partial"),
            )

        def submit_message(
            user_input: str,
            current_agent_state: dict[str, Any] | None,
            uploaded_image: str,
            chat_history: list[dict[str, str]],
            debug_log: list[str],
            model_context: list[dict[str, Any]],
            session_config: dict[str, Any],
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
                    "Session closed. Start a new session to continue.",
                    gr.update(value=""),
                    gr.update(interactive=False),
                )

            if not uploaded_image:
                return (
                    current_agent_state,
                    chat_history,
                    debug_log,
                    model_context,
                    session_closed,
                    gr.update(value=chat_history),
                    _format_debug_log(debug_log),
                    _format_model_context(model_context),
                    "No uploaded image is available for this session.",
                    gr.update(value=""),
                    gr.update(interactive=False),
                )

            agent = create_agent(session_config)

            try:
                initial_state = current_agent_state or agent.get_initial_state(
                    image_path=uploaded_image
                )
            except OSError as exc:
                next_debug_log = debug_log + [f"image_error: {exc}"]
                return (
                    current_agent_state,
                    chat_history,
                    next_debug_log,
                    model_context,
                    session_closed,
                    gr.update(value=chat_history),
                    _format_debug_log(next_debug_log),
                    _format_model_context(model_context),
                    str(exc),
                    gr.update(value=""),
                    gr.update(interactive=False),
                )

            result = agent.run_turn(initial_state, user_input)
            next_history = result["full_history"]
            next_debug_log = debug_log + result["debug_log"]
            next_model_context = result["model_context"]

            if result["session_closed"] and (
                not next_history or next_history[-1]["role"] != "assistant"
            ):
                next_history = next_history + [
                    {"role": "assistant", "content": "Session closed."}
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

        def quit_to_setup(
            session_config: dict[str, Any],
            advanced_visible: bool,
        ):
            restored_config = build_restored_config(session_config)
            base_outputs = [
                None,
                "",
                [],
                [],
                [],
                False,
                gr.update(visible=show_start_screen),
                gr.update(visible=advanced_visible),
                gr.update(visible=not show_start_screen),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=[]),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value="Ready."),
            ]

            if show_start_screen:
                return tuple(base_outputs + [
                    gr.update(value=restored_config.model),
                    gr.update(value=restored_config.api_key),
                    gr.update(value=restored_config.api_key_env),
                    gr.update(value=restored_config.base_url),
                    gr.update(value=restored_config.system_prompt),
                    gr.update(value=restored_config.trim_strategy),
                    gr.update(value=restored_config.token_counter),
                    gr.update(value=restored_config.max_tokens),
                    gr.update(value=restored_config.start_on),
                    gr.update(value=restored_config.include_system),
                    gr.update(value=restored_config.allow_partial),
                    gr.update(value=None, interactive=True),
                    gr.update(value="Upload an image to continue."),
                    gr.update(interactive=False),
                    gr.update(value=""),
                    gr.update(interactive=False),
                ])

            return tuple(base_outputs + [
                gr.update(value=None, interactive=True),
                gr.update(value="Upload an image to continue."),
                gr.update(interactive=False),
                gr.update(value=""),
                gr.update(interactive=False),
            ])

        def close_app():
            threading.Timer(0.2, app.close).start()

        if show_start_screen:
            close_button.click(fn=close_app, inputs=[], outputs=[])
            advanced_button.click(
                fn=toggle_advanced,
                inputs=[advanced_visible_state],
                outputs=[advanced_visible_state, advanced_panel],
            )
            saved_agents_radio.change(
                fn=populate_saved_agent,
                inputs=[saved_agents_radio],
                outputs=[
                    start_model,
                    start_api_key,
                    start_api_key_env,
                    start_base_url,
                    start_system_prompt,
                    start_trim_strategy,
                    start_token_counter,
                    start_max_tokens,
                    start_start_on,
                    start_include_system,
                    start_allow_partial,
                ],
            )
            start_button.click(
                fn=go_to_upload,
                inputs=[
                    start_model,
                    start_api_key,
                    start_api_key_env,
                    start_base_url,
                    start_system_prompt,
                    start_trim_strategy,
                    start_token_counter,
                    start_max_tokens,
                    start_start_on,
                    start_include_system,
                    start_allow_partial,
                ],
                outputs=[
                    session_config_state,
                    agent_state,
                    uploaded_image_state,
                    chat_history_state,
                    debug_log_state,
                    model_context_state,
                    session_closed_state,
                    internals_visible_state,
                    start_screen,
                    advanced_panel,
                    upload_screen,
                    chat_screen,
                    internals_panel,
                    image_input,
                    upload_status,
                    upload_next_button,
                    chatbot,
                    debug_output,
                    context_output,
                    status_output,
                    message_box,
                    send_button,
                ],
            )
            upload_back_button.click(
                fn=back_to_start,
                inputs=[advanced_visible_state],
                outputs=[
                    agent_state,
                    uploaded_image_state,
                    chat_history_state,
                    debug_log_state,
                    model_context_state,
                    session_closed_state,
                    internals_visible_state,
                    start_screen,
                    advanced_panel,
                    upload_screen,
                    chat_screen,
                    internals_panel,
                    image_input,
                    upload_status,
                    upload_next_button,
                    chatbot,
                    debug_output,
                    context_output,
                    status_output,
                    message_box,
                    send_button,
                ],
            )

        image_input.change(
            fn=update_upload_button,
            inputs=[image_input],
            outputs=[upload_next_button, upload_status],
        )
        upload_next_button.click(
            fn=open_chat,
            inputs=[image_input],
            outputs=[
                uploaded_image_state,
                session_closed_state,
                chat_screen,
                upload_status,
                image_input,
                upload_next_button,
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
                uploaded_image_state,
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
            fn=quit_to_setup,
            inputs=[session_config_state, advanced_visible_state],
            outputs=(
                [
                    agent_state,
                    uploaded_image_state,
                    chat_history_state,
                    debug_log_state,
                    model_context_state,
                    session_closed_state,
                    start_screen,
                    advanced_panel,
                    upload_screen,
                    chat_screen,
                    internals_panel,
                    chatbot,
                    debug_output,
                    context_output,
                    status_output,
                ]
                + (
                    [
                        start_model,
                        start_api_key,
                        start_api_key_env,
                        start_base_url,
                        start_system_prompt,
                        start_trim_strategy,
                        start_token_counter,
                        start_max_tokens,
                        start_start_on,
                        start_include_system,
                        start_allow_partial,
                        image_input,
                        upload_status,
                        upload_next_button,
                        message_box,
                        send_button,
                    ]
                    if show_start_screen
                    else [
                        image_input,
                        upload_status,
                        upload_next_button,
                        message_box,
                        send_button,
                    ]
                )
            ),
        )
        internals_button.click(
            fn=toggle_internals,
            inputs=[internals_visible_state],
            outputs=[internals_visible_state, internals_panel],
        )

    return app


def _format_debug_log(debug_log: list[str]) -> str:
    if not debug_log:
        return "No verbose events yet."
    return "\n".join(debug_log)


def _format_model_context(model_context: list[dict[str, Any]]) -> str:
    if not model_context:
        return json.dumps([], indent=2)
    return json.dumps(model_context, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the vision-language Gradio LangGraph chatbot demo."
    )
    parser.add_argument(
        "--mode",
        choices=("start", "direct"),
        default="start",
        help="Use the start screen or launch directly into image upload.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
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
    launch_kwargs = {"share": args.share}

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
        notice = "No saved agent configs were found. Using the built-in default agent."
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
            "model": agent_payload.get("model", "gpt-5.2"),
            "base_url": agent_payload.get("base_url"),
            "api_key": agent_payload.get("api_key"),
            "api_key_env": agent_payload.get("api_key_env", "OPENAI_API_KEY"),
            "system_prompt": agent_payload.get(
                "system_prompt", DEFAULT_SYSTEM_PROMPT
            ),
            "trim_strategy": agent_payload.get("trim_strategy", "last"),
            "token_counter": agent_payload.get("token_counter", "approximate"),
            "max_tokens": agent_payload.get("max_tokens", 16384),
            "start_on": agent_payload.get("start_on", "human"),
            "include_system": agent_payload.get("include_system", True),
            "allow_partial": agent_payload.get("allow_partial", False),
        },
    }


def session_config_from_saved_agent(saved_agent: dict) -> SessionConfig:
    agent_payload = saved_agent["agent"]
    return SessionConfig(
        model=agent_payload.get("model") or "gpt-5.2",
        api_key=agent_payload.get("api_key") or "",
        base_url=agent_payload.get("base_url") or "",
        api_key_env=agent_payload.get("api_key_env") or "",
        system_prompt=agent_payload.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        trim_strategy=agent_payload.get("trim_strategy") or "last",
        token_counter=agent_payload.get("token_counter") or "approximate",
        max_tokens=int(agent_payload.get("max_tokens", 16384)),
        start_on=agent_payload.get("start_on") or "human",
        include_system=bool(agent_payload.get("include_system", True)),
        allow_partial=bool(agent_payload.get("allow_partial", False)),
    )


def find_saved_agent(saved_agents: list[dict], name: str) -> SessionConfig | None:
    for saved_agent in saved_agents:
        if saved_agent["name"] == name:
            return session_config_from_saved_agent(saved_agent)
    return None


if __name__ == "__main__":
    main()
