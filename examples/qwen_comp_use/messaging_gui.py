"""
Messaging GUI - Simple chat interface for agent communication
Positions on left half of screen automatically
Agent types in this window like any other application
"""

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import webbrowser
import threading
import time
import socket

app = Flask(__name__)
app.config['SECRET_KEY'] = 'agentbox-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Message history (for new connections)
message_history = []
MAX_HISTORY = 100


def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.route('/')
def index():
    """Serve chat interface"""
    return render_template('chat.html')


@socketio.on('connect')
def handle_connect():
    """New client connected - determine role and send history"""
    # Check if local connection (agent) or remote (human)
    remote_addr = request.environ.get('REMOTE_ADDR', '')
    
    if remote_addr in ['127.0.0.1', 'localhost', '::1']:
        role = 'agent'
        print(f"[GUI] Agent connected: {request.sid}")
    else:
        role = 'human'
        print(f"[GUI] Human connected: {request.sid} from {remote_addr}")
    
    # Tell client their role
    emit('set_role', {'role': role})
    
    # Send message history
    for msg in message_history:
        emit('message', msg)


@socketio.on('user_message')
def handle_user_message(data):
    """User sent a message"""
    # Get sender's role from data
    sender_role = data.get('role', 'human')  # Default to human if not specified
    
    message = {
        'sender': sender_role,  # 'agent' or 'human'
        'text': data['text'],
        'timestamp': time.time()
    }
    
    # Store in history
    message_history.append(message)
    if len(message_history) > MAX_HISTORY:
        message_history.pop(0)
    
    # Broadcast to all clients
    emit('message', message, broadcast=True)
    
    print(f"[GUI] {sender_role.capitalize()}: {data['text']}")


@socketio.on('agent_message')
def handle_agent_message(data):
    """Agent sent a message (legacy handler - shouldn't be used)"""
    message = {
        'sender': 'agent',
        'text': data['text'],
        'timestamp': time.time()
    }
    
    # Store in history
    message_history.append(message)
    if len(message_history) > MAX_HISTORY:
        message_history.pop(0)
    
    # Broadcast to all clients
    emit('message', message, broadcast=True)
    
    print(f"[GUI] Agent: {data['text']}")


def create_templates():
    """Create HTML template"""
    import os
    os.makedirs('templates', exist_ok=True)
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Agent Messages</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: #1a1a1a;
            color: #ffffff;
        }
        
        #header {
            background: #2a2a2a;
            padding: 15px 20px;
            border-bottom: 2px solid #00ff00;
            flex-shrink: 0;
        }
        
        #header h1 {
            font-size: 20px;
            color: #00ff00;
            margin: 0;
        }
        
        #role-indicator {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        
        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .message {
            padding: 12px 16px;
            border-radius: 8px;
            max-width: 80%;
            word-wrap: break-word;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.human {
            background: #0066cc;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        
        .message.agent {
            background: #2a2a2a;
            border: 1px solid #00ff00;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        
        .message.system {
            background: #444;
            align-self: center;
            font-size: 13px;
            font-style: italic;
            color: #aaa;
        }
        
        .sender {
            font-weight: bold;
            font-size: 12px;
            margin-bottom: 4px;
            opacity: 0.8;
        }
        
        .message.human .sender {
            color: #e6f2ff;
        }
        
        .message.agent .sender {
            color: #00ff00;
        }
        
        #input-area {
            background: #2a2a2a;
            border-top: 2px solid #00ff00;
            padding: 15px 20px;
            display: flex;
            gap: 10px;
            flex-shrink: 0;
        }
        
        #message-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #1a1a1a;
            color: #ffffff;
            font-size: 14px;
            font-family: inherit;
        }
        
        #message-input:focus {
            outline: none;
            border-color: #00ff00;
        }
        
        #send-button {
            padding: 12px 24px;
            background: #00ff00;
            color: #000;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        
        #send-button:hover {
            background: #00cc00;
        }
        
        #send-button:active {
            background: #009900;
        }
        
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>🤖 Agent Communication</h1>
        <div id="role-indicator">Connecting...</div>
    </div>
    
    <div id="messages"></div>
    
    <div id="input-area">
        <input type="text" id="message-input" placeholder="Type message..." autofocus>
        <button id="send-button">Send</button>
    </div>
    
    <script>
        const socket = io();
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const roleIndicator = document.getElementById('role-indicator');
        
        // Store this client's role
        let myRole = null;
        
        // Position window on left half of screen
        window.addEventListener('load', () => {
            const screenWidth = window.screen.width;
            const screenHeight = window.screen.height;
            
            // Left half
            window.moveTo(0, 0);
            window.resizeTo(screenWidth / 2, screenHeight);
            
            console.log('Window positioned on left half');
        });
        
        // Add system message
        function addSystemMessage(text) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message system';
            messageDiv.textContent = text;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Add message to chat
        function addMessage(sender, text) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const senderLabel = document.createElement('div');
            senderLabel.className = 'sender';
            senderLabel.textContent = sender === 'human' ? 'Human' : 'Agent';
            
            const textDiv = document.createElement('div');
            textDiv.textContent = text;
            
            messageDiv.appendChild(senderLabel);
            messageDiv.appendChild(textDiv);
            messagesDiv.appendChild(messageDiv);
            
            // Auto-scroll to bottom
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Send message
        function sendMessage() {
            const text = messageInput.value.trim();
            if (text && myRole) {
                socket.emit('user_message', { text: text, role: myRole });
                messageInput.value = '';
            }
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Socket events
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            roleIndicator.textContent = 'Disconnected';
            addSystemMessage('Disconnected from server');
        });
        
        socket.on('set_role', (data) => {
            myRole = data.role;
            console.log('My role:', myRole);
            
            // Update header
            if (myRole === 'agent') {
                roleIndicator.textContent = 'You are: Agent (local window)';
                roleIndicator.style.color = '#00ff00';
                addSystemMessage('Agent window ready - Type here to communicate');
            } else {
                roleIndicator.textContent = 'You are: Human (remote connection)';
                roleIndicator.style.color = '#0066cc';
                addSystemMessage('Connected - Send messages to the agent');
            }
        });
        
        socket.on('message', (data) => {
            addMessage(data.sender, data.text);
        });
    </script>
</body>
</html>"""
    
    with open('templates/chat.html', 'w') as f:
        f.write(html)


def open_browser(url, delay=2):
    """Open browser after delay"""
    time.sleep(delay)
    print(f"[GUI] Opening browser: {url}")
    webbrowser.open(url)


def run_server(host='0.0.0.0', port=5000, open_browser_window=True):
    """Run the messaging GUI server"""
    
    # Create templates
    create_templates()
    
    # Get local IP
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("  Agent Messaging GUI Server")
    print("="*60)
    print(f"\nLocal access:    http://localhost:{port}")
    print(f"Network access:  http://{local_ip}:{port}")
    print(f"\nConnect from other computer: http://{local_ip}:{port}")
    print("\nAgent will type in this window like any other application.")
    print("="*60 + "\n")
    
    # Open browser in separate thread
    if open_browser_window:
        threading.Thread(
            target=open_browser, 
            args=(f"http://localhost:{port}",),
            daemon=True
        ).start()
    
    # Run server
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(open_browser_window=True)
