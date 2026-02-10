"""
Computer Use Agent with ReAct Framework
Uses official computer_use tool interface
Agent must use action tool to interact with messaging GUI
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageGrab
import pyautogui
import json
import re
import time
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
import http.server
import socketserver
import threading


@dataclass
class ContextMessage:
    """Message in agent's context window"""
    type: str  # 'screenshot', 'thought', 'action', 'result'
    content: str
    screenshot: Optional[Image.Image] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ComputerUseAgent:
    """
    Computer control agent using official tool interface
    ReAct loop: Screenshot → Thought → Action → repeat
    """
    
    def __init__(
        self,
        task_description: str,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_context: int = 25,
        debug_port: Optional[int] = 8080
    ):
        self.task_description = task_description
        self.max_context = max_context
        self.debug_port = debug_port
        
        print(f"\n[Agent] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("[Agent] Model loaded!")
        
        # Context window (rolling buffer)
        self.context: List[ContextMessage] = []
        
        # System context (stable, set once)
        self.system_context = f"""You are controlling a computer. The user will tell you the goal. You will receive screenshots depicting your desktop. On the left should be a messaging interface for communicating with the user. On the right should be your task workspace.

User goal: {task_description}"""
        
        # Tool definition (official format from training)
        self.computer_tool = self._build_tool_definition()
        
        # State
        self.running = False
        self.step_count = 0
        
        # Screen config
        screen_size = pyautogui.size()
        self.screen_width = screen_size.width
        self.screen_height = screen_size.height
        
        # Coordinate normalization (0-1000, as in training)
        self.norm_width = 1000
        self.norm_height = 1000
        
        # Temp directory for screenshots
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        
        # Start debug server if requested
        if self.debug_port:
            self._start_debug_server()
    
    def _build_tool_definition(self) -> Dict:
        """Build official computer_use tool definition"""
        return {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": """Use mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots.
* The screen's resolution is normalized to 1000x1000 coordinate space.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element.
* The LEFT side of screen shows a messaging app - you can type there to communicate with the user.
* The RIGHT side shows your task area - interact there to complete your goal.""",
                "parameters": {
                    "type": "object",
                    "properties": {
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
                            "description": "[x, y] in 0-1000 normalized space",
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
                    "required": ["action"]
                }
            }
        }
    
    def _start_debug_server(self):
        """Start HTTP server for context inspection"""
        class ContextHandler(http.server.SimpleHTTPRequestHandler):
            agent = None
            
            def do_GET(self):
                if self.path == '/context':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    html = self._build_context_html()
                    self.wfile.write(html.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _build_context_html(self):
                """Build HTML showing agent context"""
                html = ["<html><head><title>Agent Context</title>"]
                html.append("<meta http-equiv='refresh' content='3'>")  # Auto-refresh
                html.append("<style>")
                html.append("body{font-family:monospace;background:#1a1a1a;color:#0f0;padding:20px;margin:0;}")
                html.append("h1{color:#0f0;border-bottom:2px solid #0f0;padding-bottom:10px;}")
                html.append("h2{color:#0a0;margin-top:30px;}")
                html.append(".info{color:#888;margin:10px 0;}")
                html.append(".context-item{margin:15px 0;padding:15px;border:1px solid #333;border-radius:5px;background:#0a0a0a;}")
                html.append(".type{color:#00f;font-weight:bold;}")
                html.append(".observation{border-left:3px solid #00f;}")
                html.append(".thought{border-left:3px solid #0f0;}")
                html.append(".action{border-left:3px solid #f80;}")
                html.append(".result{border-left:3px solid #f0f;}")
                html.append(".content{margin-top:8px;color:#ccc;white-space:pre-wrap;word-wrap:break-word;}")
                html.append(".timestamp{color:#666;font-size:0.9em;}")
                html.append(".system-context{background:#002200;padding:15px;border:1px solid #0f0;border-radius:5px;margin:20px 0;}")
                html.append("</style></head><body>")
                
                html.append(f"<h1>🤖 Agent Debug Console</h1>")
                html.append(f"<div class='info'>Step: {ContextHandler.agent.step_count} | ")
                html.append(f"Context Size: {len(ContextHandler.agent.context)}/{ContextHandler.agent.max_context}</div>")
                
                # System Context
                html.append("<div class='system-context'>")
                html.append("<h2>System Context</h2>")
                html.append(f"<pre>{ContextHandler.agent.system_context}</pre>")
                html.append("</div>")
                
                # Context History
                html.append("<h2>Context History (Last {}):</h2>".format(
                    min(len(ContextHandler.agent.context), ContextHandler.agent.max_context)
                ))
                
                if not ContextHandler.agent.context:
                    html.append("<div class='info'>No context yet</div>")
                else:
                    for i, msg in enumerate(ContextHandler.agent.context[-ContextHandler.agent.max_context:]):
                        html.append(f"<div class='context-item {msg.type}'>")
                        html.append(f"<span class='type'>{msg.type.upper()}</span> ")
                        html.append(f"<span class='timestamp'>{datetime.fromtimestamp(msg.timestamp).strftime('%H:%M:%S')}</span>")
                        html.append(f"<div class='content'>{msg.content}</div>")
                        html.append("</div>")
                
                html.append("</body></html>")
                return "".join(html)
            
            def log_message(self, format, *args):
                pass  # Suppress request logs
        
        ContextHandler.agent = self
        
        def serve():
            # Bind to 0.0.0.0 so it's accessible from other computers
            with socketserver.TCPServer(("0.0.0.0", self.debug_port), ContextHandler) as httpd:
                # Get local IP for display
                import socket
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                except Exception:
                    local_ip = "127.0.0.1"
                
                print(f"[Agent] Debug server: http://localhost:{self.debug_port}/context")
                print(f"[Agent] Network access: http://{local_ip}:{self.debug_port}/context")
                httpd.serve_forever()
        
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
    
    def capture_screenshot(self) -> Image.Image:
        """Capture full screen"""
        screenshot = ImageGrab.grab()
        # Resize for model
        screenshot = screenshot.resize((1280, 720))
        return screenshot
    
    def normalize_coordinates(self, norm_x: float, norm_y: float) -> tuple:
        """Convert normalized (0-1000) to screen pixels"""
        x = int(norm_x / self.norm_width * self.screen_width)
        y = int(norm_y / self.norm_height * self.screen_height)
        return x, y
    
    def add_to_context(self, msg_type: str, content: str, screenshot: Optional[Image.Image] = None):
        """Add message to context window"""
        msg = ContextMessage(
            type=msg_type,
            content=content,
            screenshot=screenshot
        )
        self.context.append(msg)
        
        # Keep only last N messages
        if len(self.context) > self.max_context:
            self.context.pop(0)
    
    def build_prompt(self, screenshot: Image.Image, prompt_for: str = "thought") -> str:
        """Build prompt with clean structure: System Context + Context History + Phase Prompt"""
        
        # Part 1: System Context (stable)
        prompt_parts = [self.system_context, ""]
        
        # Part 2: Context History (last 25 messages)
        if self.context:
            prompt_parts.append("CONTEXT HISTORY:")
            for msg in self.context[-self.max_context:]:
                prompt_parts.append(f"{msg.type}: {msg.content}")
            prompt_parts.append("")
        
        # Part 3: Phase Prompt (alternates)
        if prompt_for == "thought":
            phase_prompt = "Enter a thought; What do you see, what should be done next to accomplish the user goal, have you made any mistakes or was the last action successful."
        elif prompt_for == "action":
            phase_prompt = "Use the action tool to control the computer."
        else:
            phase_prompt = ""
        
        prompt_parts.append(phase_prompt)
        
        return "\n".join(prompt_parts)
    
    def generate_response(self, screenshot: Image.Image, prompt: str) -> str:
        """Generate response from model"""
        
        # Save screenshot to temp file for qwen_agent's ContentItem
        import os
        screenshot_path = os.path.join(self.temp_dir, f"screenshot_{self.step_count}.png")
        screenshot.save(screenshot_path)
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": screenshot}
                ]
            }
        ]
        
        # Add tools only when asking for action
        if "Use the action tool" in prompt:
            # Apply chat template with tools
            from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
                NousFnCallPrompt, Message, ContentItem
            )
            
            fn_prompt = NousFnCallPrompt()
            qwen_messages = [
                Message(role="user", content=[
                    ContentItem(text=prompt),
                    ContentItem(image=f"file://{screenshot_path}")  # Use file path, not PIL Image
                ])
            ]
            
            processed = fn_prompt.preprocess_fncall_messages(
                messages=qwen_messages,
                functions=[self.computer_tool],
                lang=None
            )
            
            messages = [msg.model_dump() for msg in processed]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[screenshot],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return response
    
    def parse_action(self, response: str) -> Optional[Dict]:
        """Parse action from response"""
        
        # Look for <tool_call>...</tool_call>
        match = re.search(r'<tool_call>\s*(\{.+?\})\s*</tool_call>', 
                         response, re.DOTALL)
        
        if match:
            try:
                tool_data = json.loads(match.group(1))
                return tool_data.get("arguments", {})
            except json.JSONDecodeError:
                return None
        
        return None
    
    def execute_action(self, action_dict: Dict) -> str:
        """Execute action using pyautogui"""
        
        action = action_dict.get("action")
        
        if action == "screenshot":
            return "Screenshot captured"
        
        elif action == "mouse_move":
            coord = action_dict.get("coordinate", [500, 500])
            x, y = self.normalize_coordinates(coord[0], coord[1])
            pyautogui.moveTo(x, y, duration=0.3)
            return f"Moved mouse to ({x}, {y})"
        
        elif action == "left_click":
            coord = action_dict.get("coordinate")
            if coord:
                x, y = self.normalize_coordinates(coord[0], coord[1])
                pyautogui.click(x, y)
                return f"Clicked at ({x}, {y})"
            else:
                pyautogui.click()
                return "Clicked at current position"
        
        elif action == "right_click":
            coord = action_dict.get("coordinate")
            if coord:
                x, y = self.normalize_coordinates(coord[0], coord[1])
                pyautogui.rightClick(x, y)
                return f"Right-clicked at ({x}, {y})"
            else:
                pyautogui.rightClick()
                return "Right-clicked at current position"
        
        elif action == "double_click":
            coord = action_dict.get("coordinate")
            if coord:
                x, y = self.normalize_coordinates(coord[0], coord[1])
                pyautogui.doubleClick(x, y)
                return f"Double-clicked at ({x}, {y})"
            else:
                pyautogui.doubleClick()
                return "Double-clicked at current position"
        
        elif action == "type":
            text = action_dict.get("text", "")
            pyautogui.write(text, interval=0.05)
            return f"Typed: {text}"
        
        elif action == "key":
            keys = action_dict.get("keys", [])
            pyautogui.hotkey(*keys)
            return f"Pressed: {'+'.join(keys)}"
        
        elif action == "scroll":
            pixels = action_dict.get("pixels", 0)
            pyautogui.scroll(int(pixels))
            return f"Scrolled {pixels} pixels"
        
        elif action == "wait":
            duration = action_dict.get("time", 1)
            time.sleep(duration)
            return f"Waited {duration} seconds"
        
        elif action == "terminate":
            status = action_dict.get("status", "success")
            return f"Task terminated: {status}"
        
        elif action == "fail":
            message = action_dict.get("message", "Unknown error")
            return f"Task failed: {message}"
        
        else:
            return f"Unknown action: {action}"
    
    def step(self) -> bool:
        """
        Single ReAct step: Screenshot → Thought → Action
        Returns True if should continue, False if done
        """
        
        self.step_count += 1
        print(f"\n{'='*60}")
        print(f"STEP {self.step_count}")
        print(f"{'='*60}\n")
        
        # 1. Capture screenshot
        print("[1/3] Capturing screenshot...")
        screenshot = self.capture_screenshot()
        self.add_to_context("screenshot", "New screenshot captured", screenshot)
        
        # 2. Generate thought
        print("[2/3] Generating thought...")
        thought_prompt = self.build_prompt(screenshot, prompt_for="thought")
        thought = self.generate_response(screenshot, thought_prompt)
        
        # Clean thought (remove any tool calls if present)
        thought = re.sub(r'<tool_call>.*?</tool_call>', '', thought, flags=re.DOTALL).strip()
        
        self.add_to_context("thought", thought)
        print(f"[Thought] {thought}\n")
        
        # 3. Generate and execute action
        print("[3/3] Generating action...")
        action_prompt = self.build_prompt(screenshot, prompt_for="action")
        action_response = self.generate_response(screenshot, action_prompt)
        
        # Parse action
        action_dict = self.parse_action(action_response)
        
        if action_dict:
            action_name = action_dict.get("action")
            print(f"[Action] {action_name}: {action_dict}")
            
            self.add_to_context("action", f"{action_name} - {json.dumps(action_dict)}")
            
            # Execute
            result = self.execute_action(action_dict)
            self.add_to_context("result", result)
            print(f"[Result] {result}")
            
            # Check if done
            if action_name in ["terminate", "fail"]:
                print(f"\n{'='*60}")
                print(f"AGENT FINISHED: {result}")
                print(f"{'='*60}\n")
                return False
            
            return True
        else:
            print("[Action] No action parsed - treating as completion")
            return False
    
    def run(self, step_delay: float = 2.0):
        """Run agent loop"""
        
        print("\n" + "="*60)
        print("  AGENT STARTING")
        print("="*60)
        print(f"\nTask: {self.task_description}")
        print(f"Max context: {self.max_context} messages")
        print(f"Step delay: {step_delay}s")
        if self.debug_port:
            # Get local IP for network access
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
                
                # Delay between steps
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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python computer_agent.py 'Your task description'")
        print("Example: python computer_agent.py 'Complete the certificate training quiz'")
        sys.exit(1)
    
    task = " ".join(sys.argv[1:])
    
    agent = ComputerUseAgent(
        task_description=task,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        max_context=25,
        debug_port=8080
    )
    
    agent.run(step_delay=2.0)
