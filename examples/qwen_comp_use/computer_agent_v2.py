"""
Computer Use Agent - Clean Implementation
Follows design documented in computer_agent.md

Key principles:
- LangChain messages with proper roles (system/user/assistant/tool)
- Single generation per step (thought + action together)
- Custom adaptation layers for local HF Qwen (documented)
- No manual prompt building
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from PIL import Image, ImageGrab
import pyautogui
import json
import re
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict
import tempfile
import os
import http.server
import socketserver
import threading


class ComputerUseAgent:
    """
    Computer control agent using vision-language model with proper message handling.
    
    See computer_agent.md for complete design documentation.
    """
    
    def __init__(
        self,
        task_description: str,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_context_pairs: int = 12,
        debug_port: Optional[int] = 8080
    ):
        """
        Initialize agent with task.
        
        Args:
            task_description: User's goal (e.g., "Complete the quiz")
            model_name: Hugging Face model ID
            max_context_pairs: How many screenshot-action-result cycles to keep
            debug_port: Port for debug web interface (None to disable)
        """
        self.task_description = task_description
        self.max_context_pairs = max_context_pairs
        self.debug_port = debug_port
        
        # Load model
        print(f"\n[Agent] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("[Agent] Model loaded!")
        
        # Initialize message history (LangChain format)
        # See computer_agent.md section "1. Separate System and User Goal Messages"
        self.messages: List[BaseMessage] = [
            SystemMessage(content=self._build_system_message()),
            HumanMessage(content=f"User goal: {task_description}")
        ]
        
        # Tool definition with thought parameter
        # See computer_agent.md section "2. Thought Inside Tool Call"
        self.computer_tool = self._build_tool_definition()
        
        # State
        self.running = False
        self.step_count = 0
        
        # Screen config
        screen_size = pyautogui.size()
        self.screen_width = screen_size.width
        self.screen_height = screen_size.height
        print(f"[Agent] Screen size: {self.screen_width}x{self.screen_height}")
        
        # Coordinate normalization (Qwen training format: 0-1000)
        self.norm_width = 1000
        self.norm_height = 1000
        
        # Temp directory for screenshot storage
        # Needed for LangChain image_url format
        self.temp_dir = tempfile.mkdtemp()
        
        # Start debug server
        if self.debug_port:
            self._start_debug_server()
    
    def _build_system_message(self) -> str:
        """Build system message explaining agent's role and tool usage."""
        return """You are controlling a computer. You will receive screenshots depicting your desktop.

LAYOUT:
- LEFT side: Messaging interface for communicating with the user
- RIGHT side: Your task workspace

IMPORTANT: You MUST ALWAYS respond using the computer_use tool. Never respond with plain text.

The tool requires:
- thought: Your reasoning (what you see, what to do next, any mistakes)
- action: The computer action to take (click, type, etc.)

Example:
{
  "thought": "I see a quiz question asking about X. I should click option B because...",
  "action": "left_click",
  "coordinate": [750, 400]
}

Always use the tool to take actions. Explain your reasoning in the thought field."""
    
    def _build_tool_definition(self) -> Dict:
        """
        Build tool definition with thought parameter.
        
        See computer_agent.md section "2. Thought Inside Tool Call"
        This combines ReAct-style reasoning with action in one call.
        """
        return {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": "Control the computer with mouse and keyboard. Always include your reasoning in 'thought' before taking action.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your reasoning: what you see on screen, what needs to be done next to accomplish the user goal, any mistakes from previous actions, or whether the goal is complete"
                        },
                        "action": {
                            "type": "string",
                            "enum": [
                                "mouse_move",
                                "left_click",
                                "right_click",
                                "double_click",
                                "type",
                                "key",
                                "scroll",
                                "wait",
                                "screenshot",
                                "terminate",
                                "fail"
                            ],
                            "description": "The action to perform"
                        },
                        "coordinate": {
                            "type": "array",
                            "description": "[x, y] coordinates in 0-1000 normalized space",
                            "items": {"type": "number"}
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type (for action=type)"
                        },
                        "keys": {
                            "type": "array",
                            "description": "Keys to press (for action=key)",
                            "items": {"type": "string"}
                        },
                        "pixels": {
                            "type": "number",
                            "description": "Scroll amount (for action=scroll)"
                        },
                        "time": {
                            "type": "number",
                            "description": "Duration in seconds (for action=wait)"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["success", "failure"],
                            "description": "Task status (for action=terminate/fail)"
                        },
                        "message": {
                            "type": "string",
                            "description": "Explanation (for action=fail)"
                        }
                    },
                    "required": ["thought", "action"]
                }
            }
        }
    
    def capture_screenshot(self) -> Image.Image:
        """Capture current screen state."""
        screenshot = ImageGrab.grab()
        # Resize for model efficiency
        screenshot = screenshot.resize((1280, 720))
        return screenshot
    
    def _save_screenshot(self, screenshot: Image.Image) -> str:
        """
        Save screenshot to temp file and return path.
        
        Why: LangChain's image_url format expects a URL or file path.
        See computer_agent.md section "3. Image Handling"
        """
        filename = f"screenshot_{self.step_count}.png"
        filepath = os.path.join(self.temp_dir, filename)
        screenshot.save(filepath)
        return filepath
    
    def _extract_images_from_messages(self, messages: List[BaseMessage]) -> List[Image.Image]:
        """
        Extract PIL images from messages for HF processor.
        
        Why: HF processor needs images as separate list, not in message content.
        This is our "adaptation layer" for local HF models.
        See computer_agent.md section "3. Image Handling"
        """
        images = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        # Extract from our temp file storage
                        url = item["image_url"]["url"]
                        if url.startswith("file://"):
                            filepath = url[7:]  # Remove "file://" prefix
                            if os.path.exists(filepath):
                                images.append(Image.open(filepath))
        return images
    
    def trim_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Keep system context + recent interactions.
        
        Strategy:
        - Always keep: SystemMessage (index 0)
        - Always keep: User goal (index 1)
        - Keep last N triplets: (HumanMessage + AIMessage + ToolMessage)
        
        See computer_agent.md section "7. Context Window Management"
        """
        if len(messages) <= 2:
            return messages
        
        # System and user goal
        system_msgs = messages[:2]
        
        # Recent interactions
        recent = messages[2:]
        
        # Keep last max_context_pairs * 3 messages
        # (each pair = HumanMessage with screenshot + AIMessage with tool_call + ToolMessage with result)
        max_recent = self.max_context_pairs * 3
        trimmed_recent = recent[-max_recent:] if len(recent) > max_recent else recent
        
        return system_msgs + trimmed_recent
    
    def generate_response(self) -> AIMessage:
        """
        Generate model response as AIMessage with tool calls.
        
        Process:
        1. Trim context to prevent overflow
        2. Convert messages to dicts (for apply_chat_template)
        3. Apply chat template (adds special tokens, formats roles)
        4. Extract images separately (HF processor requirement)
        5. Generate
        6. Parse tool call from response
        7. Return as AIMessage
        
        See computer_agent.md section "8. Message Formatting for Model"
        """
        # 1. Trim context
        trimmed = self.trim_messages(self.messages)
        
        # 2. Convert to dicts for apply_chat_template
        # Note: We use LangChain's .dict() method which includes role info
        message_dicts = [msg.dict() for msg in trimmed]
        
        # 3. Apply chat template with tools
        # This is where Qwen formats everything with special tokens
        # CRITICAL: This is model-specific formatting - don't manually build prompts
        from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
            NousFnCallPrompt, Message, ContentItem
        )
        
        fn_prompt = NousFnCallPrompt()
        
        # Convert to qwen_agent format
        # Note: This is an adaptation layer because we're using local HF
        # See computer_agent.md section "4. Tool Call Parsing"
        qwen_messages = []
        for msg_dict in message_dicts:
            role = msg_dict["role"]
            content = msg_dict["content"]
            
            if role == "system":
                qwen_messages.append(Message(role="system", content=[ContentItem(text=content)]))
            elif role == "user":
                if isinstance(content, str):
                    qwen_messages.append(Message(role="user", content=[ContentItem(text=content)]))
                elif isinstance(content, list):
                    # Handle multi-part content (text + image)
                    items = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                items.append(ContentItem(text=item["text"]))
                            elif item.get("type") == "image_url":
                                # Use file path for qwen_agent
                                url = item["image_url"]["url"]
                                items.append(ContentItem(image=url))
                    qwen_messages.append(Message(role="user", content=items))
            elif role == "assistant":
                # Skip tool_calls in preprocessing - they're already in history
                qwen_messages.append(Message(role="assistant", content=[ContentItem(text=content or "")]))
            elif role == "tool":
                # Tool results
                qwen_messages.append(Message(role="tool", content=[ContentItem(text=content)]))
        
        # Preprocess with tool definition
        processed = fn_prompt.preprocess_fncall_messages(
            messages=qwen_messages,
            functions=[self.computer_tool],
            lang=None
        )
        
        final_messages = [msg.model_dump() for msg in processed]
        
        # 4. Apply chat template (final formatting)
        text = self.processor.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 5. Extract images for processor
        images = self._extract_images_from_messages(trimmed)
        
        # 6. Generate
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # 7. Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # 8. Parse tool call
        # See computer_agent.md section "4. Tool Call Parsing"
        tool_call = self._parse_tool_call(response_text)
        
        if tool_call:
            return AIMessage(content="", tool_calls=[tool_call])
        else:
            # Model didn't use tool - this violates system prompt
            # See computer_agent.md section "6. Handling Non-Tool Responses"
            return AIMessage(content=response_text)
    
    def _parse_tool_call(self, response_text: str) -> Optional[Dict]:
        """
        Parse Qwen's XML tool call format into LangChain format.
        
        Input (from model):
            <tool_call>
            {"name": "computer_use", "arguments": {"thought": "...", "action": "left_click", ...}}
            </tool_call>
        
        Output (for LangChain):
            {
                "name": "computer_use",
                "args": {"thought": "...", "action": "left_click", ...},
                "id": "call_abc123"
            }
        
        See computer_agent.md section "4. Tool Call Parsing"
        """
        # Look for <tool_call>...</tool_call>
        match = re.search(r'<tool_call>\s*(\{.+?\})\s*</tool_call>', 
                         response_text, re.DOTALL)
        
        if not match:
            return None
        
        try:
            # Parse JSON
            tool_data = json.loads(match.group(1))
            
            # Convert to LangChain format
            return {
                "name": tool_data.get("name"),
                "args": tool_data.get("arguments", {}),
                "id": f"call_{uuid.uuid4().hex[:8]}"
            }
        except json.JSONDecodeError as e:
            print(f"[Warning] Failed to parse tool JSON: {e}")
            print(f"Raw: {match.group(1)[:200]}")
            return None
    
    def normalize_coordinates(self, norm_x: float, norm_y: float) -> tuple:
        """
        Convert normalized (0-1000) coordinates to absolute screen pixels.
        
        Why 0-1000: Qwen was trained with this normalization.
        See computer_agent.md (coordinate normalization section)
        """
        abs_x = int(norm_x / self.norm_width * self.screen_width)
        abs_y = int(norm_y / self.norm_height * self.screen_height)
        
        # Bounds check
        abs_x = max(0, min(abs_x, self.screen_width - 1))
        abs_y = max(0, min(abs_y, self.screen_height - 1))
        
        return abs_x, abs_y
    
    def execute_action(self, args: Dict) -> str:
        """Execute computer action using pyautogui."""
        action = args.get("action")
        
        if action == "screenshot":
            return "Screenshot captured"
        
        elif action == "mouse_move":
            coord = args.get("coordinate", [500, 500])
            abs_x, abs_y = self.normalize_coordinates(coord[0], coord[1])
            pyautogui.moveTo(abs_x, abs_y, duration=0.3)
            return f"Moved mouse to ({abs_x}, {abs_y})"
        
        elif action == "left_click":
            coord = args.get("coordinate")
            if coord:
                abs_x, abs_y = self.normalize_coordinates(coord[0], coord[1])
                pyautogui.click(abs_x, abs_y)
                return f"Clicked at ({abs_x}, {abs_y})"
            else:
                pyautogui.click()
                return "Clicked at current position"
        
        elif action == "right_click":
            coord = args.get("coordinate")
            if coord:
                abs_x, abs_y = self.normalize_coordinates(coord[0], coord[1])
                pyautogui.rightClick(abs_x, abs_y)
                return f"Right-clicked at ({abs_x}, {abs_y})"
            else:
                pyautogui.rightClick()
                return "Right-clicked"
        
        elif action == "double_click":
            coord = args.get("coordinate")
            if coord:
                abs_x, abs_y = self.normalize_coordinates(coord[0], coord[1])
                pyautogui.doubleClick(abs_x, abs_y)
                return f"Double-clicked at ({abs_x}, {abs_y})"
            else:
                pyautogui.doubleClick()
                return "Double-clicked"
        
        elif action == "type":
            text = args.get("text", "")
            pyautogui.write(text, interval=0.05)
            return f"Typed: {text}"
        
        elif action == "key":
            keys = args.get("keys", [])
            pyautogui.hotkey(*keys)
            return f"Pressed: {'+'.join(keys)}"
        
        elif action == "scroll":
            pixels = args.get("pixels", 0)
            pyautogui.scroll(int(pixels))
            return f"Scrolled {pixels} pixels"
        
        elif action == "wait":
            duration = args.get("time", 1)
            time.sleep(duration)
            return f"Waited {duration} seconds"
        
        elif action == "terminate":
            status = args.get("status", "success")
            return f"Task {status}"
        
        elif action == "fail":
            message = args.get("message", "Unknown error")
            return f"Task failed: {message}"
        
        else:
            return f"Unknown action: {action}"
    
    def step(self) -> bool:
        """
        Single agent step: Screenshot → Generate → Execute.
        
        Returns True to continue, False to stop.
        
        See computer_agent.md section "5. Single Generation Per Step"
        """
        self.step_count += 1
        print(f"\n{'='*60}")
        print(f"STEP {self.step_count}")
        print(f"{'='*60}\n")
        
        # 1. Capture screenshot
        print("[1/3] Capturing screenshot...")
        screenshot = self.capture_screenshot()
        screenshot_path = self._save_screenshot(screenshot)
        
        # Add to message history in LangChain format
        self.messages.append(HumanMessage(content=[
            {"type": "text", "text": "Current screenshot"},
            {"type": "image_url", "image_url": {"url": f"file://{screenshot_path}"}}
        ]))
        
        # 2. Generate (thought + action in one call)
        print("[2/3] Generating response...")
        ai_message = self.generate_response()
        
        # Check if model used tool
        if not ai_message.tool_calls:
            # Model violated system prompt - didn't use tool
            # See computer_agent.md section "6. Handling Non-Tool Responses"
            print(f"[Warning] Model didn't use tool!")
            print(f"Response: {ai_message.content[:200]}")
            print("[Warning] Not adding to history - continuing...")
            return True  # Try again
        
        # Add AI message to history
        self.messages.append(ai_message)
        
        # Display thought and action
        tool_call = ai_message.tool_calls[0]
        args = tool_call['args']
        thought = args.get('thought', 'No thought provided')
        action = args.get('action', 'unknown')
        
        print(f"[Thought] {thought}\n")
        print(f"[Action] {action}: {args}")
        
        # 3. Execute action
        result = self.execute_action(args)
        print(f"[Result] {result}")
        
        # Add result to history as ToolMessage
        self.messages.append(ToolMessage(
            content=result,
            tool_call_id=tool_call['id']
        ))
        
        # Check if done
        if action in ["terminate", "fail"]:
            print(f"\n{'='*60}")
            print(f"AGENT FINISHED: {result}")
            print(f"{'='*60}\n")
            return False
        
        return True
    
    def run(self, step_delay: float = 2.0):
        """Run agent loop."""
        
        print("\n" + "="*60)
        print("  AGENT STARTING")
        print("="*60)
        print(f"\nTask: {self.task_description}")
        print(f"Screen: {self.screen_width}x{self.screen_height}")
        print(f"Max context: {self.max_context_pairs} interaction pairs")
        print(f"Step delay: {step_delay}s")
        if self.debug_port:
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "127.0.0.1"
            
            print(f"Debug local: http://localhost:{self.debug_port}/context")
            print(f"Debug network: http://{local_ip}:{self.debug_port}/context")
        print("\n" + "="*60 + "\n")
        
        print("⚠️  WARNING: Agent will control mouse and keyboard in 3 seconds!")
        for i in range(3, 0, -1):
            print(f"   Starting in {i}...")
            time.sleep(1)
        print()
        
        self.running = True
        
        try:
            while self.running:
                should_continue = self.step()
                
                if not should_continue:
                    break
                
                time.sleep(step_delay)
        
        except KeyboardInterrupt:
            print("\n\n[Agent] Stopped by user (Ctrl+C)")
        
        except Exception as e:
            print(f"\n\n[Agent] Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup temp directory
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
            
            print("\n[Agent] Shutdown complete")
    
    def _start_debug_server(self):
        """Start HTTP debug server for inspecting agent state."""
        class DebugHandler(http.server.SimpleHTTPRequestHandler):
            agent = None
            
            def do_GET(self):
                if self.path == '/context':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html = self._build_html()
                    self.wfile.write(html.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _build_html(self):
                html = ["<html><head><title>Agent Context</title>"]
                html.append("<meta http-equiv='refresh' content='3'>")
                html.append("<style>")
                html.append("body{font-family:monospace;background:#1a1a1a;color:#0f0;padding:20px;}")
                html.append("h1{color:#0f0;border-bottom:2px solid #0f0;padding-bottom:10px;}")
                html.append(".info{color:#888;margin:10px 0;}")
                html.append(".msg{margin:15px 0;padding:15px;border:1px solid #333;border-radius:5px;background:#0a0a0a;}")
                html.append(".role{color:#0ff;font-weight:bold;}")
                html.append(".system{border-left:3px solid #00f;}")
                html.append(".user{border-left:3px solid #0f0;}")
                html.append(".assistant{border-left:3px solid #f80;}")
                html.append(".tool{border-left:3px solid #f0f;}")
                html.append(".content{margin-top:8px;color:#ccc;white-space:pre-wrap;}")
                html.append("</style></head><body>")
                
                html.append(f"<h1>🤖 Agent Debug Console</h1>")
                html.append(f"<div class='info'>Step: {DebugHandler.agent.step_count} | ")
                html.append(f"Messages: {len(DebugHandler.agent.messages)}</div>")
                
                html.append("<h2>Message History:</h2>")
                
                for i, msg in enumerate(DebugHandler.agent.messages):
                    role = type(msg).__name__.replace("Message", "").lower()
                    html.append(f"<div class='msg {role}'>")
                    html.append(f"<span class='role'>{role.upper()}</span>")
                    
                    # Content
                    if isinstance(msg.content, str):
                        html.append(f"<div class='content'>{msg.content[:500]}</div>")
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    html.append(f"<div class='content'>{item['text'][:500]}</div>")
                                elif item.get("type") == "image_url":
                                    html.append(f"<div class='content'>[Image]</div>")
                    
                    # Tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            html.append(f"<div class='content'><strong>Tool:</strong> {tc['name']}</div>")
                            if 'thought' in tc['args']:
                                html.append(f"<div class='content'><strong>Thought:</strong> {tc['args']['thought']}</div>")
                            html.append(f"<div class='content'><strong>Action:</strong> {tc['args'].get('action')}</div>")
                    
                    html.append("</div>")
                
                html.append("</body></html>")
                return "".join(html)
            
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        DebugHandler.agent = self
        
        def serve():
            with socketserver.TCPServer(("0.0.0.0", self.debug_port), DebugHandler) as httpd:
                httpd.serve_forever()
        
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python computer_agent.py 'Your task description'")
        print("Example: python computer_agent.py 'Complete the quiz'")
        sys.exit(1)
    
    task = " ".join(sys.argv[1:])
    
    agent = ComputerUseAgent(
        task_description=task,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        max_context_pairs=12,
        debug_port=8080
    )
    
    agent.run(step_delay=2.0)
