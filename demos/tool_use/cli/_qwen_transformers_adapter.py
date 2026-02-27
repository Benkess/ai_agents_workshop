import json
import os
import re
import uuid
from typing import Any

import torch
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
TOOL_OUTPUT_CHAR_LIMIT = 4000


def _resolve_dtype(dtype_str: str | None, device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if dtype_str == "bfloat16":
            return torch.bfloat16
        if dtype_str == "float16":
            return torch.float16
        if dtype_str == "float32":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _truncate_text(text: str, limit: int = TOOL_OUTPUT_CHAR_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[truncated]"


def _build_tool_specs(tools: list[BaseTool]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for tool in tools:
        parameters = {"type": "object", "properties": {}, "required": []}
        if getattr(tool, "args_schema", None) is not None:
            schema_method = getattr(tool.args_schema, "model_json_schema", None)
            if callable(schema_method):
                parameters = schema_method()
            else:
                schema_method = getattr(tool.args_schema, "schema", None)
                if callable(schema_method):
                    parameters = schema_method()

        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )
    return specs


def _serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for tool_call in tool_calls:
        payload = {
            "name": tool_call.get("name"),
            "arguments": tool_call.get("args", {}),
        }
        blocks.append(f"<tool_call>{json.dumps(payload, ensure_ascii=True)}</tool_call>")
    return "\n".join(blocks)


def _convert_message_to_qwen_dict(message: AnyMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": str(message.content)}

    if isinstance(message, HumanMessage):
        return {"role": "user", "content": str(message.content)}

    if isinstance(message, ToolMessage):
        tool_name = getattr(message, "name", None) or "tool_result"
        content = (
            f"tool_call_id={message.tool_call_id}\n"
            f"tool_name={tool_name}\n"
            f"result:\n{_truncate_text(str(message.content))}"
        )
        return {"role": "function", "name": tool_name, "content": content}

    if isinstance(message, AIMessage):
        content = str(message.content or "")
        if message.tool_calls:
            tool_call_markup = _serialize_tool_calls(message.tool_calls)
            content = f"{content}\n{tool_call_markup}".strip()
        return {"role": "assistant", "content": content}

    return {"role": "user", "content": str(getattr(message, "content", ""))}


def _build_prompt_messages(messages: list[AnyMessage], tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_instruction = (
        "You are a helpful assistant with access to tools.\n"
        "If a tool can answer faster or more reliably, call it.\n"
        "When calling tools, emit one or more <tool_call>{JSON}</tool_call> blocks and nothing else for that turn.\n"
        "Each JSON object must have the shape "
        '{"name": "<tool_name>", "arguments": {...}}.\n'
        "When no tool is needed, respond normally.\n"
        f"Available tools:\n{json.dumps(tool_specs, ensure_ascii=True, indent=2)}"
    )

    prompt_messages: list[dict[str, Any]] = [{"role": "system", "content": tool_instruction}]
    prompt_messages.extend(_convert_message_to_qwen_dict(message) for message in messages)
    return prompt_messages


def _safe_parse_tool_call_json(raw_json: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if "name" not in data or "arguments" not in data:
        return None

    arguments = data["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None

    return {"name": data["name"], "arguments": arguments}


def _extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    tool_calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    cursor = 0

    for match in TOOL_CALL_BLOCK_RE.finditer(text):
        start, end = match.span()
        raw_block = match.group(0)
        raw_json = match.group(1)
        content_parts.append(text[cursor:start])

        parsed = _safe_parse_tool_call_json(raw_json)
        if parsed is None:
            content_parts.append(raw_block)
        else:
            tool_calls.append(
                {
                    "name": parsed["name"],
                    "args": parsed["arguments"],
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
            )
        cursor = end

    content_parts.append(text[cursor:])
    content = "".join(content_parts).strip()
    return content, tool_calls


class QwenTransformersAdapter:
    def __init__(
        self,
        tools: list[BaseTool],
        model_name: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        requested_device = device or os.getenv("QWEN_DEVICE")
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"

        requested_dtype = dtype or os.getenv("QWEN_DTYPE")
        resolved_dtype = _resolve_dtype(requested_dtype, requested_device)

        self.model_name = model_name or os.getenv("QWEN_MODEL", DEFAULT_MODEL_NAME)
        self.device = requested_device
        self.dtype = resolved_dtype
        self.max_new_tokens = max_new_tokens
        self.tool_specs = _build_tool_specs(tools)

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        base_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
        }

        if self.device.startswith("cuda"):
            base_kwargs["device_map"] = "auto"

        model = None
        try:
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, **base_kwargs)
        except Exception:
            try:
                model = AutoModelForVision2Seq.from_pretrained(self.model_name, **base_kwargs)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **base_kwargs)

        if not self.device.startswith("cuda"):
            model.to(self.device)
        return model

    def invoke(self, messages: list[AnyMessage]) -> AIMessage:
        prompt_messages = _build_prompt_messages(messages, self.tool_specs)
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=[prompt_text], padding=True, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids = []
        for input_ids, full_output_ids in zip(inputs["input_ids"], output_ids):
            generated_ids.append(full_output_ids[len(input_ids) :])

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        content, tool_calls = _extract_tool_calls(text)
        return AIMessage(content=content, tool_calls=tool_calls)
