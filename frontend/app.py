"""
Enhanced Dash Frontend for Enterprise RAG System with Logo Integration
UPDATED: Added logo support in top-left corner
"""
import dash
import os
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from datetime import datetime
import requests
import json
import base64
import io
import uuid
import traceback
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    assets_folder='assets'  # Tell Dash where to find static assets
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Function to encode logo as base64 for embedding
def get_logo_base64():
    """Get logo as base64 string for embedding in HTML"""
    logo_path = Path(__file__).parent / "assets" / "logo.svg"
    
    if logo_path.exists():
        try:
            with open(logo_path, 'rb') as f:
                logo_data = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/svg+xml;base64,{logo_data}"
        except Exception as e:
            logger.error(f"Error loading logo: {e}")
            return None
    else:
        logger.warning(f"Logo not found at: {logo_path}")
        return None

# Enhanced CSS with logo styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0f0f0f;
                font-family: 'Inter', sans-serif;
            }
            
            .app-container {
                display: flex;
                height: 100vh;
                background: #0f0f0f;
            }
            
            .sidebar {
                width: 280px;
                background: #1a1a1a;
                border-right: 1px solid #2a2a2a;
                display: flex;
                flex-direction: column;
                padding: 20px;
            }
            
            .sidebar-header {
                display: flex;
                align-items: center;
                margin-bottom: 25px;
                padding-bottom: 20px;
                border-bottom: 1px solid #2a2a2a;
            }
            
            .logo-container {
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .logo-image {
                width: 80px;
                height: 80px;
                flex-shrink: 0;
            }
            
            .logo-text {
                color: #fff;
                font-size: 20px;
                font-weight: 600;
                margin: 0;
                line-height: 1.2;
            }
            
            .section-title {
                color: #ccc;
                font-size: 14px;
                font-weight: 500;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: #0f0f0f;
            }
            
            .chat-header {
                padding: 20px 30px;
                border-bottom: 1px solid #2a2a2a;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #1a1a1a;
            }
            
            .chat-title {
                color: #fff;
                font-size: 20px;
                font-weight: 600;
                margin: 0;
            }
            
            .messages-container {
                flex: 1;
                overflow-y: auto;
                padding: 20px 30px;
                background: #0f0f0f;
                scroll-behavior: smooth;
            }
            
            .input-area {
                padding: 20px 30px;
                background: #1a1a1a;
                border-top: 1px solid #2a2a2a;
            }
            
            .input-wrapper {
                display: flex;
                gap: 10px;
                align-items: flex-end;
            }
            
            .message-input {
                flex: 1;
                background: #2a2a2a !important;
                border: 1px solid #3a3a3a !important;
                color: white !important;
                border-radius: 8px;
                padding: 12px;
                resize: none;
            }
            
            .message-input:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            .send-btn {
                height: 48px;
                width: 48px;
                border-radius: 8px;
                transition: all 0.2s;
            }
            
            .send-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .message-wrapper {
                margin-bottom: 20px;
                animation: fadeIn 0.3s ease-in;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .user-wrapper {
                display: flex;
                justify-content: flex-end;
            }
            
            .assistant-wrapper {
                display: flex;
                justify-content: flex-start;
            }
            
            .message-bubble {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 12px;
                word-wrap: break-word;
            }
            
            .user-message {
                background: #2563eb;
                color: white;
            }
            
            .assistant-message {
                background: #2a2a2a;
                color: #e0e0e0;
            }
            
            .assistant-message.thinking {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
            }
            
            .typing-indicator {
                display: inline-flex;
                gap: 4px;
                padding: 8px 12px;
                align-items: center;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #666;
                animation: typing 1.4s infinite;
            }
            
            .typing-dot:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-dot:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                    background: #666;
                }
                30% {
                    transform: translateY(-10px);
                    background: #2563eb;
                }
            }
            
            .upload-area {
                border: 2px dashed #3a3a3a;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                color: #888;
                margin-top: 10px;
            }
            
            .upload-area:hover {
                border-color: #2563eb;
                background: rgba(37, 99, 235, 0.1);
            }
            
            .status-dot {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 8px;
            }
            
            .status-dot.online {
                background-color: #10b981;
            }
            
            .status-dot.offline {
                background-color: #ef4444;
            }
            
            .session-info {
                background: #2a2a2a;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 10px;
                font-size: 12px;
                color: #888;
            }
            
            .mcp-tools {
                background: #2a2a2a;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            
            .mcp-tool-toggle {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 8px;
                border-radius: 6px;
                margin-bottom: 5px;
                transition: all 0.2s;
            }
            
            .mcp-tool-toggle:hover {
                background: #3a3a3a;
            }
            
            .mcp-tool-toggle label {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #ccc;
                cursor: pointer;
                font-size: 14px;
            }
            
            .mcp-tool-icon {
                width: 20px;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .welcome-message {
                text-align: center;
                padding: 40px;
                color: #888;
            }
            
            .welcome-message h2 {
                color: #fff;
                margin-bottom: 10px;
            }
            
            .welcome-message p {
                margin-bottom: 20px;
            }
            
            .quick-actions {
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            .quick-action-btn {
                padding: 8px 16px;
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                color: #ccc;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .quick-action-btn:hover {
                background: #3a3a3a;
                color: #fff;
            }
            
            .actions-section {
                margin-bottom: 20px;
            }
            
            .upload-section {
                margin-bottom: 20px;
            }
            
            .upload-status {
                margin-top: 10px;
                font-size: 12px;
            }
            
            .sidebar-footer {
                margin-top: auto;
                padding-top: 15px;
                border-top: 1px solid #2a2a2a;
            }
            
            .status-bar {
                display: flex;
                align-items: center;
                font-size: 12px;
                color: #888;
            }
            
            .status-text {
                font-weight: 500;
            }
            
            .action-btn {
                margin-bottom: 8px !important;
            }
            
            /* Auto-scroll to bottom */
            .messages-container {
                scroll-behavior: smooth;
            }
        </style>
        <script>
            // Auto-scroll to bottom of messages
            function scrollToBottom() {
                const container = document.querySelector('.messages-container');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            }
            
            // Enhanced Enter key handling
            document.addEventListener('DOMContentLoaded', function() {
                // Set up Enter key listener
                setInterval(function() {
                    const textarea = document.getElementById('message-input');
                    if (textarea && !textarea.hasListener) {
                        textarea.hasListener = true;
                        textarea.addEventListener('keydown', function(e) {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                const sendBtn = document.getElementById('send-btn');
                                if (sendBtn && !sendBtn.disabled) {
                                    sendBtn.click();
                                }
                            }
                        });
                    }
                }, 100);
                
                // Auto-scroll on new messages
                const observer = new MutationObserver(function(mutations) {
                    scrollToBottom();
                });
                
                const container = document.querySelector('.messages-container');
                if (container) {
                    observer.observe(container, { childList: true, subtree: true });
                }
            });
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create logo component
def create_logo_component():
    """Create the logo component for the sidebar header"""
    logo_base64 = get_logo_base64()
    
    if logo_base64:
        # Logo found - create image element
        logo_element = html.Img(
            src=logo_base64,
            className="logo-image",
            alt="Enterprise RAG Logo"
        )
    else:
        # Fallback icon if logo not found
        logo_element = html.I(
            className="fas fa-brain",
            style={
                "fontSize": "48px",
                "color": "#2563eb",
                "width": "48px",
                "height": "48px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            }
        )
    
    return html.Div([
        logo_element,
        html.H4("Enterprise RAG", className="logo-text"),
    ], className="logo-container")

# Layout with updated sidebar header
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "dark"},
    children=[
        html.Div([
            # Sidebar
            html.Div([
                # Logo/Title Header - UPDATED
                html.Div([
                    create_logo_component()
                ], className="sidebar-header"),
                
                # Session Info
                html.Div(id="session-info", className="session-info"),
                
                # MCP Tools Section
                html.Div([
                    html.Div("🔧 MCP Tools", className="section-title"),
                    html.Div([
                        # Sequential Thinking Toggle
                        html.Div([
                            html.Label([
                                html.Span(className="mcp-tool-icon"),
                                html.Span("Sequential Thinking"),
                                dcc.Checklist(
                                    id="sequential-thinking-toggle",
                                    options=[{"label": "", "value": "enabled"}],
                                    value=["enabled"],
                                    style={"marginLeft": "auto", "display": "inline-block"}
                                )
                            ], style={"display": "flex", "alignItems": "center", "width": "100%"})
                        ], className="mcp-tool-toggle"),
                        
                        # Web Search Toggle
                        html.Div([
                            html.Label([
                                html.Span(className="mcp-tool-icon"),
                                html.Span("Web Search"),
                                dcc.Checklist(
                                    id="web-search-toggle",
                                    options=[{"label": "", "value": "enabled"}],
                                    value=[],
                                    style={"marginLeft": "auto", "display": "inline-block"}
                                )
                            ], style={"display": "flex", "alignItems": "center", "width": "100%"})
                        ], className="mcp-tool-toggle"),
                    ], className="mcp-tools"),
                ], style={"marginBottom": "20px"}),
                
                # Actions
                html.Div([
                    html.Div("Actions", className="section-title"),
                    dbc.Button(
                        [html.I(className="fas fa-broom"), " Clear Chat"],
                        id="clear-chat-btn",
                        className="action-btn",
                        color="secondary",
                        size="sm",
                        style={"width": "100%", "marginBottom": "10px"}
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-plus"), " New Session"],
                        id="new-session-btn",
                        className="action-btn",
                        color="primary",
                        size="sm",
                        style={"width": "100%", "marginBottom": "10px"}
                    ),
                ], className="actions-section"),
                
                # Upload Section
                html.Div([
                    html.Div("📁 Documents", className="section-title"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-upload"),
                            ' Drop files or click to upload'
                        ]),
                        className="upload-area",
                        multiple=True
                    ),
                    html.Div(id="upload-status", className="upload-status"),
                ], className="upload-section"),
                
                # Status
                html.Div([
                    html.Div([
                        html.Span("●", id="status-indicator", className="status-dot offline"),
                        html.Span("Connecting...", id="status-text", className="status-text"),
                    ], className="status-bar"),
                    html.Div(id="session-stats", style={"fontSize": "11px", "color": "#666", "marginTop": "10px"}),
                ], className="sidebar-footer"),
            ], className="sidebar"),
            
            # Main Content Area
            html.Div([
                # Chat Header
                html.Div([
                    html.H5(id="chat-title", children="Chat Assistant", className="chat-title"),
                    html.Div([
                        html.Span(id="upload-processing", style={"color": "#888", "fontSize": "14px"}),
                    ], className="chat-actions"),
                ], className="chat-header"),
                
                # Messages Area
                html.Div(id="messages-container", className="messages-container"),
                
                # Input Area
                html.Div([
                    html.Div([
                        dbc.Textarea(
                            id="message-input",
                            placeholder="Ask a question... (Press Enter to send, Shift+Enter for new line)",
                            className="message-input",
                            style={"minHeight": "50px", "maxHeight": "150px"},
                            disabled=False
                        ),
                        dbc.Button(
                            html.I(className="fas fa-paper-plane"),
                            id="send-btn",
                            className="send-btn",
                            color="primary",
                            disabled=False
                        ),
                    ], className="input-wrapper"),
                    
                    # Processing Status
                    html.Div(id="processing-status", className="processing-status"),
                ], className="input-area"),
            ], className="main-content"),
        ], className="app-container"),
        
        # Hidden stores
        dcc.Store(id="session-store", data={}),
        dcc.Store(id="current-session-id", data=None),
        dcc.Store(id="upload-trigger", data=0),
        dcc.Store(id="pending-message", data=None),
        dcc.Store(id="message-sent-trigger", data=0),
        dcc.Interval(id="status-interval", interval=10000),  # Check status every 10s
        dcc.Interval(id="session-stats-interval", interval=30000),  # Update stats every 5s
        dcc.Interval(id="init-interval", interval=100, max_intervals=1),  # One-time init
    ]
)

# Initialize session on page load
@app.callback(
    [Output("current-session-id", "data"),
     Output("messages-container", "children"),
     Output("session-info", "children")],
    Input("init-interval", "n_intervals"),
    State("current-session-id", "data"),
    prevent_initial_call=False
)
def initialize_session(n, current_session_id):
    """Initialize session automatically on page load"""
    if current_session_id:
        return no_update, no_update, no_update
    
    try:
        # Create initial session
        response = requests.post(f"{API_URL}/sessions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            logger.info(f"Auto-created session: {session_id}")
            
            session_info = html.Div([
                html.Div(f"Session: {session_id[:8]}...", style={"fontWeight": "bold"}),
                html.Div("Documents: 0"),
                html.Div("Chunks: 0"),
            ])
            
            # Welcome message
            welcome = html.Div([
                html.H2("👋 Welcome to Enterprise RAG"),
                html.P("I'm ready to help you analyze your documents."),
                html.Div([
                    html.Span("Upload documents using the sidebar, or start chatting right away!"),
                ], className="quick-actions")
            ], className="welcome-message")
            
            return session_id, [welcome], session_info
    except Exception as e:
        logger.error(f"Failed to auto-create session: {e}")
    
    # Fallback to local session
    session_id = str(uuid.uuid4())
    session_info = html.Div([
        html.Div(f"Session: {session_id[:8]}... (local)", style={"fontWeight": "bold"}),
    ])
    
    welcome = html.Div([
        html.H2("👋 Welcome to Enterprise RAG"),
        html.P("Working in offline mode."),
    ], className="welcome-message")
    
    return session_id, [welcome], session_info

# Clear chat messages only
@app.callback(
    Output("messages-container", "children", allow_duplicate=True),
    Input("clear-chat-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_chat_messages(n_clicks):
    """Clear chat messages without affecting session"""
    if not n_clicks:
        raise PreventUpdate
    
    welcome = html.Div([
        html.H2("💬 Chat Cleared"),
        html.P("Your documents are still available. Start a new conversation!"),
    ], className="welcome-message")
    
    return [welcome]

# Create new session
@app.callback(
    [Output("current-session-id", "data", allow_duplicate=True),
     Output("messages-container", "children", allow_duplicate=True),
     Output("upload-status", "children"),
     Output("session-info", "children", allow_duplicate=True)],
    Input("new-session-btn", "n_clicks"),
    State("current-session-id", "data"),
    prevent_initial_call=True
)
def create_new_session(n_clicks, current_id):
    """Create a new session and cleanup old one"""
    if not n_clicks:
        raise PreventUpdate
    
    # Cleanup previous session if exists
    if current_id:
        try:
            cleanup_response = requests.post(
                f"{API_URL}/sessions/{current_id}/cleanup", 
                timeout=10
            )
            logger.info(f"Cleaned up previous session: {current_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup session: {e}")
    
    # Create new session
    try:
        response = requests.post(f"{API_URL}/sessions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            new_session_id = data.get("session_id")
            
            session_info = html.Div([
                html.Div(f"Session: {new_session_id[:8]}...", style={"fontWeight": "bold"}),
                html.Div("Documents: 0"),
                html.Div("Chunks: 0"),
            ])
            
            welcome = html.Div([
                html.H2("🆕 New Session Started"),
                html.P("Ready for new documents and conversations!"),
            ], className="welcome-message")
            
            return new_session_id, [welcome], html.Div("New session ready", style={"color": "#10b981"}), session_info
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
    
    return no_update, no_update, no_update, no_update

# Improved message sending with immediate UI feedback
@app.callback(
    [Output("messages-container", "children", allow_duplicate=True),
     Output("message-input", "value"),
     Output("message-input", "disabled"),
     Output("send-btn", "disabled"),
     Output("pending-message", "data")],
    [Input("send-btn", "n_clicks")],
    [State("message-input", "value"),
     State("messages-container", "children"),
     State("current-session-id", "data"),
     State("sequential-thinking-toggle", "value"),
     State("web-search-toggle", "value")],
    prevent_initial_call=True
)
def send_message_immediate(n_clicks, message, current_messages, session_id, seq_thinking, web_search):
    """Immediately show user message and typing indicator"""
    if not n_clicks or not message or not message.strip():
        raise PreventUpdate
    
    if not session_id:
        return current_messages, "", False, False, None
    
    # Initialize messages if None
    if current_messages is None or (len(current_messages) == 1 and "welcome-message" in str(current_messages[0])):
        current_messages = []
    
    # Add user message immediately
    user_msg = create_message_bubble(message, "user")
    current_messages.append(user_msg)
    
    # Add typing indicator
    typing_indicator = html.Div([
        html.Div([
            html.Div([
                html.Span("Thinking", style={"marginRight": "8px", "color": "#888"}),
                html.Span(className="typing-dot"),
                html.Span(className="typing-dot"),
                html.Span(className="typing-dot"),
            ], className="typing-indicator")
        ], className="message-bubble assistant-message thinking")
    ], className="message-wrapper assistant-wrapper", id="typing-indicator")
    
    current_messages.append(typing_indicator)
    
    # Store message data for processing
    message_data = {
        "text": message,
        "session_id": session_id,
        "seq_thinking": seq_thinking,
        "web_search": web_search
    }
    
    return current_messages, "", True, True, message_data

# Process the actual query in background
@app.callback(
    [Output("messages-container", "children", allow_duplicate=True),
     Output("message-input", "disabled", allow_duplicate=True),
     Output("send-btn", "disabled", allow_duplicate=True)],
    Input("pending-message", "data"),
    State("messages-container", "children"),
    prevent_initial_call=True
)
def process_message(message_data, current_messages):
    """Process the message and get response"""
    if not message_data:
        raise PreventUpdate
    
    message = message_data["text"]
    session_id = message_data["session_id"]
    seq_thinking = message_data.get("seq_thinking", [])
    web_search = message_data.get("web_search", [])
    
    # Remove typing indicator
    if current_messages and len(current_messages) > 0:
        # Check if last message is typing indicator
        if "typing-indicator" in str(current_messages[-1]):
            current_messages = current_messages[:-1]
    
    try:
        # Prepare the query with MCP tools
        enhanced_query = message
        
        # Add MCP tool instructions if enabled
        tool_instructions = []
        if "enabled" in seq_thinking:
            tool_instructions.append("Use sequential thinking to break down the problem step by step.")
        if "enabled" in web_search:
            tool_instructions.append("Search the web for current information if needed.")
        
        if tool_instructions:
            enhanced_query = f"{message}\n\n[Tools: {', '.join(tool_instructions)}]"
        
        # Query the API
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": enhanced_query,
                "session_id": session_id,
                "retrieval_k": 50,
                "use_reranker": True
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No response generated")
            sources = data.get("source_documents", [])
            
            # Create assistant message with any tool indicators
            if "enabled" in seq_thinking or "enabled" in web_search:
                tools_used = []
                if "enabled" in seq_thinking:
                    tools_used.append("🧠 Sequential Thinking")
                if "enabled" in web_search:
                    tools_used.append("🔍 Web Search")
                
                answer = f"[{' • '.join(tools_used)}]\n\n{answer}"
            
            assistant_msg = create_message_bubble(answer, "assistant", sources)
            current_messages.append(assistant_msg)
            
        else:
            error_msg = create_message_bubble(
                "Sorry, I encountered an error processing your request. Please try again.",
                "assistant"
            )
            current_messages.append(error_msg)
            
    except Exception as e:
        logger.error(f"Query error: {e}")
        error_msg = create_message_bubble(
            "Connection error. Please check if the server is running.",
            "assistant"
        )
        current_messages.append(error_msg)
    
    # Re-enable input
    return current_messages, False, False

# File upload handler
@app.callback(
    [Output("upload-status", "children", allow_duplicate=True),
     Output("upload-trigger", "data")],
    Input("upload-data", "contents"),
    [State("upload-data", "filename"),
     State("upload-trigger", "data"),
     State("current-session-id", "data")],
    prevent_initial_call=True
)
def upload_files(contents, filenames, trigger, session_id):
    """Handle file uploads with session isolation"""
    if contents is None:
        raise PreventUpdate
    
    if not session_id:
        return [html.Div("Please wait for session initialization...", style={"color": "#f59e0b"})], trigger
    
    status_messages = []
    
    try:
        for content, filename in zip(contents, filenames):
            try:
                # Parse the content
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                
                # Create file object for upload
                files = {'files': (filename, io.BytesIO(decoded), 'application/octet-stream')}
                
                # Upload to session-specific endpoint
                upload_response = requests.post(
                    f"{API_URL}/upload/{session_id}",
                    files=files,
                    timeout=30
                )
                
                if upload_response.status_code == 200:
                    upload_data = upload_response.json()
                    
                    # Index the uploaded files
                    index_response = requests.post(
                        f"{API_URL}/index",
                        json={
                            "file_paths": upload_data.get("file_paths", []),
                            "session_id": session_id,
                            "index_name": f"session_{session_id}"
                        },
                        timeout=60
                    )
                    
                    if index_response.status_code == 200:
                        index_data = index_response.json()
                        status_messages.append(
                            html.Div([
                                html.I(className="fas fa-check-circle", style={"color": "#10b981"}),
                                f" {filename} ({index_data.get('chunks_created', 0)} chunks)"
                            ], style={"color": "#10b981", "margin": "5px 0"})
                        )
                    else:
                        status_messages.append(
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"color": "#f59e0b"}),
                                f" {filename} uploaded but indexing failed"
                            ], style={"color": "#f59e0b", "margin": "5px 0"})
                        )
                else:
                    status_messages.append(
                        html.Div([
                            html.I(className="fas fa-times-circle", style={"color": "#ef4444"}),
                            f" {filename} failed to upload"
                        ], style={"color": "#ef4444", "margin": "5px 0"})
                    )
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                status_messages.append(
                    html.Div([
                        html.I(className="fas fa-exclamation-circle", style={"color": "#ef4444"}),
                        f" {filename} error"
                    ], style={"color": "#ef4444", "margin": "5px 0"})
                )
    
    except Exception as e:
        logger.error(f"Upload handler error: {e}")
        status_messages = [
            html.Div([
                html.I(className="fas fa-exclamation-circle", style={"color": "#ef4444"}),
                f" Upload failed"
            ], style={"color": "#ef4444"})
        ]
    
    return html.Div(status_messages), trigger + 1

# Status check callback
@app.callback(
    [Output("status-indicator", "className"),
     Output("status-text", "children")],
    Input("status-interval", "n_intervals"),
    prevent_initial_call=False
)
def update_status(n):
    """Check API connection status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=30)
        if response.status_code == 200:
            return "status-dot online", "Connected"
    except:
        pass
    return "status-dot offline", "Disconnected"
     
# Session stats update
@app.callback(
    [Output("session-stats", "children"),
     Output("session-info", "children", allow_duplicate=True)],
    [Input("session-stats-interval", "n_intervals"),
     Input("upload-trigger", "data")],  # Also trigger on uploads
    State("current-session-id", "data"),
    prevent_initial_call=True
)
def update_session_stats(n, upload_trigger, session_id):
    """Update session statistics display"""
    if not session_id:
        return "", no_update
    
    try:
        response = requests.get(f"{API_URL}/sessions/{session_id}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            doc_count = stats.get('document_count', 0)
            chunk_count = stats.get('chunk_count', 0)
            
            # Update the bottom stats
            stats_text = f"Docs: {doc_count} | Chunks: {chunk_count}"
            
            # Update the session info in sidebar
            session_info = html.Div([
                html.Div(f"Session: {session_id[:8]}...", style={"fontWeight": "bold"}),
                html.Div(f"Documents: {doc_count}"),
                html.Div(f"Chunks: {chunk_count}"),
            ])
            
            return stats_text, session_info
    except:
        pass
    
    return "", no_update

def create_message_bubble(content, sender, sources=None, web_sources=None):
    """Create a styled message bubble with enhanced tool indicators"""
    if sender == "user":
        return html.Div([
            html.Div([
                html.Div(content, className="message-content"),
            ], className="message-bubble user-message")
        ], className="message-wrapper user-wrapper")
    else:
        message_parts = []
        
        # Check for tool indicators in content
        tools_used = []
        if "[🧠 Sequential Thinking" in content:
            tools_used.append("Sequential Thinking")
        if "[🔍 Web Search" in content:
            tools_used.append("Web Search")
        
        # Create tool usage badge if tools were used
        if tools_used:
            tool_badge = html.Div([
                html.Div([
                    html.Span("Tools Used: ", style={"fontWeight": "bold", "marginRight": "8px"}),
                    *[html.Span(
                        tool, 
                        style={
                            "backgroundColor": "#2563eb" if "Sequential" in tool else "#10b981",
                            "color": "white",
                            "padding": "2px 8px",
                            "borderRadius": "4px",
                            "marginRight": "4px",
                            "fontSize": "12px"
                        }
                    ) for tool in tools_used]
                ], style={"marginBottom": "10px"})
            ], className="tool-usage-badge")
            message_parts.append(tool_badge)
        
        # Add main message content
        message_bubble = html.Div([
            html.Div([
                html.Div(content, className="message-content", style={"whiteSpace": "pre-wrap"}),
                
                # Add web sources if available (separate from document sources)
                html.Div([
                    html.Details([
                        html.Summary("🌐 Web Search Results", className="sources-summary", 
                                   style={"cursor": "pointer", "color": "#10b981", "marginTop": "10px"}),
                        html.Div([
                            html.Div([
                                html.Strong(f"{i}. ", style={"color": "#10b981"}),
                                html.A(
                                    result.get("title", ""),
                                    href=result.get("url", "#"),
                                    target="_blank",
                                    style={"color": "#60a5fa", "textDecoration": "none"}
                                ),
                                html.P(
                                    result.get("snippet", "")[:150] + "...",
                                    style={"fontSize": "11px", "color": "#999", "marginLeft": "20px"}
                                )
                            ], className="web-source-item", style={"marginBottom": "8px"})
                            for i, result in enumerate(web_sources[:5], 1)
                        ] if web_sources else [html.Div("No web results", style={"color": "#666"})], 
                        className="web-sources-list", style={"marginTop": "10px"})
                    ], className="web-sources-section")
                ]) if web_sources else None,
                
                # Add document sources if available
                html.Div([
                    html.Details([
                        html.Summary("📚 Document Sources", className="sources-summary", 
                                   style={"cursor": "pointer", "color": "#888", "marginTop": "10px"}),
                        html.Div([
                            html.Div([
                                html.Strong(f"Source {i+1}: ", style={"color": "#2563eb"}),
                                html.Span(doc.get("text", "")[:200] + "...", 
                                        className="source-text",
                                        style={"fontSize": "12px", "color": "#aaa"})
                            ], className="source-item", style={"marginBottom": "8px"})
                            for i, doc in enumerate(sources[:3])
                        ] if sources else [html.Div("No document sources", style={"color": "#666"})], 
                        className="sources-list", style={"marginTop": "10px"})
                    ], className="sources-section")
                ]) if sources else None
            ]),
        ], className="message-bubble assistant-message")
        
        message_parts.append(message_bubble)
        
        return html.Div(message_parts, className="message-wrapper assistant-wrapper")

# Run the app
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Enhanced Enterprise RAG Frontend with Logo Starting...")
    print("=" * 60)
    print(f"📍 Frontend URL: http://localhost:3000")
    print(f"📍 API URL: {API_URL}")
    print("✨ Features:")
    print("  • Logo integration in top-left corner")
    print("  • Auto-session creation on page load")
    print("  • Instant message display with typing indicator")
    print("  • MCP Tools: Sequential Thinking & Web Search")
    print("  • Improved UX with smooth animations")
    print("=" * 60)
    print("📁 Logo path: /home/ubuntu/enterprise-rag/frontend/assets/logo.svg")
    print("💡 If logo doesn't appear, check that the file exists at the path above")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=3000)