"""
Simplified Session Manager
"""
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages chat sessions and conversation history"""
    
    def __init__(self, db_path: str = "./backend/chat_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')
        
        # Indexes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indexes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                document_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        # Session-Index link table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_indexes (
                session_id TEXT NOT NULL,
                index_id TEXT NOT NULL,
                linked_at TEXT NOT NULL,
                PRIMARY KEY (session_id, index_id),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (index_id) REFERENCES indexes(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, title: str = "New Chat") -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (id, title, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, title, now, now, '{}'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to a session"""
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_str = json.dumps(metadata or {})
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (message_id, session_id, role, content, now, metadata_str))
        
        # Update session timestamp
        cursor.execute('''
            UPDATE sessions SET updated_at = ? WHERE id = ?
        ''', (now, session_id))
        
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for a session"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM messages 
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            msg = dict(row)
            msg['metadata'] = json.loads(msg['metadata'])
            messages.append(msg)
        
        conn.close()
        return messages
    
    def get_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all sessions"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        ''', (limit,))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return sessions
    
    def create_index(self, name: str, document_count: int = 0) -> str:
        """Create a new index"""
        index_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO indexes (id, name, created_at, document_count, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (index_id, name, now, document_count, '{}'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created index: {index_id}")
        return index_id
    
    def link_index_to_session(self, session_id: str, index_id: str):
        """Link an index to a session"""
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO session_indexes (session_id, index_id, linked_at)
            VALUES (?, ?, ?)
        ''', (session_id, index_id, now))
        
        conn.commit()
        conn.close()
    
    def get_session_indexes(self, session_id: str) -> List[str]:
        """Get indexes linked to a session"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT index_id FROM session_indexes
            WHERE session_id = ?
            ORDER BY linked_at
        ''', (session_id,))
        
        index_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return index_ids
