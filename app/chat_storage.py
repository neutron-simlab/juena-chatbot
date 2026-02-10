"""
SQLite chat storage for Streamlit sidebar.

This module provides persistent storage for chat sessions and messages,
enabling chat history in the sidebar like ChatGPT.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from juena.core.config import global_config
from juena.schema.server import ChatMessage


@dataclass
class Chat:
    """Represents a chat session."""
    thread_id: str
    title: str = "New Chat"
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_summarized: bool = False
    user_message_count: int = 0


class ChatStorage:
    """SQLite-based chat storage for sidebar history."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize chat storage.
        
        Args:
            db_path: Path to SQLite database. Defaults to global_config.CHAT_DB_PATH.
        """
        self.db_path = db_path or global_config.CHAT_DB_PATH
        
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    summary TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_summarized INTEGER DEFAULT 0,
                    user_message_count INTEGER DEFAULT 0
                )
            """)
            
            # Create chat_messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT DEFAULT '[]',
                    tool_call_id TEXT,
                    run_id TEXT,
                    response_metadata TEXT DEFAULT '{}',
                    custom_data TEXT DEFAULT '{}',
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES chats(thread_id) ON DELETE CASCADE
                )
            """)
            
            # Create index for faster message lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_thread_id 
                ON chat_messages(thread_id)
            """)
            
            conn.commit()
    
    def upsert_chat(self, chat: Chat) -> None:
        """
        Insert or update a chat session.
        
        Args:
            chat: Chat object to save
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chats (thread_id, title, summary, created_at, updated_at, is_summarized, user_message_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    title = excluded.title,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at,
                    is_summarized = excluded.is_summarized,
                    user_message_count = excluded.user_message_count
            """, (
                chat.thread_id,
                chat.title,
                chat.summary,
                chat.created_at.isoformat(),
                chat.updated_at.isoformat(),
                1 if chat.is_summarized else 0,
                chat.user_message_count
            ))
            conn.commit()
    
    def get_chat(self, thread_id: str) -> Optional[Chat]:
        """
        Get a chat by thread ID.
        
        Args:
            thread_id: Thread ID to look up
            
        Returns:
            Chat object or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats WHERE thread_id = ?", (thread_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Chat(
                thread_id=row["thread_id"],
                title=row["title"],
                summary=row["summary"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                is_summarized=bool(row["is_summarized"]),
                user_message_count=row["user_message_count"]
            )
    
    def list_chats(self, limit: int = 50) -> list[Chat]:
        """
        List all chats ordered by most recently updated.
        
        Args:
            limit: Maximum number of chats to return
            
        Returns:
            List of Chat objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chats 
                ORDER BY updated_at DESC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
            return [
                Chat(
                    thread_id=row["thread_id"],
                    title=row["title"],
                    summary=row["summary"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    is_summarized=bool(row["is_summarized"]),
                    user_message_count=row["user_message_count"]
                )
                for row in rows
            ]
    
    def save_message(self, thread_id: str, message: ChatMessage) -> None:
        """
        Save a message to the database.
        
        Also updates the chat's updated_at timestamp and user_message_count.
        
        Args:
            thread_id: Thread ID the message belongs to
            message: ChatMessage to save
        """
        now = datetime.now()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert message
            cursor.execute("""
                INSERT INTO chat_messages 
                (thread_id, message_type, content, tool_calls, tool_call_id, run_id, response_metadata, custom_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                thread_id,
                message.type,
                message.content,
                json.dumps(message.tool_calls),
                message.tool_call_id,
                message.run_id,
                json.dumps(message.response_metadata),
                json.dumps(message.custom_data),
                now.isoformat()
            ))
            
            # Update chat's updated_at and increment user_message_count if human message
            if message.type == "human":
                cursor.execute("""
                    UPDATE chats SET 
                        updated_at = ?,
                        user_message_count = user_message_count + 1
                    WHERE thread_id = ?
                """, (now.isoformat(), thread_id))
            else:
                cursor.execute("""
                    UPDATE chats SET updated_at = ? WHERE thread_id = ?
                """, (now.isoformat(), thread_id))
            
            conn.commit()
    
    def load_messages(self, thread_id: str) -> list[ChatMessage]:
        """
        Load all messages for a chat.
        
        Args:
            thread_id: Thread ID to load messages for
            
        Returns:
            List of ChatMessage objects in chronological order
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chat_messages 
                WHERE thread_id = ? 
                ORDER BY id ASC
            """, (thread_id,))
            rows = cursor.fetchall()
            
            return [
                ChatMessage(
                    type=row["message_type"],
                    content=row["content"],
                    tool_calls=json.loads(row["tool_calls"]),
                    tool_call_id=row["tool_call_id"],
                    run_id=row["run_id"],
                    response_metadata=json.loads(row["response_metadata"]),
                    custom_data=json.loads(row["custom_data"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None
                )
                for row in rows
            ]
    
    def delete_chat(self, thread_id: str) -> None:
        """
        Delete a chat and all its messages.
        
        Args:
            thread_id: Thread ID to delete
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Messages are deleted via CASCADE
            cursor.execute("DELETE FROM chats WHERE thread_id = ?", (thread_id,))
            conn.commit()
    
    def get_unsummarized_chats(self, min_messages: int = 5) -> list[Chat]:
        """
        Get chats that need summarization.
        
        Args:
            min_messages: Minimum user messages required for summarization
            
        Returns:
            List of Chat objects that need summarization
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chats 
                WHERE is_summarized = 0 AND user_message_count >= ?
                ORDER BY updated_at DESC
            """, (min_messages,))
            rows = cursor.fetchall()
            
            return [
                Chat(
                    thread_id=row["thread_id"],
                    title=row["title"],
                    summary=row["summary"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    is_summarized=bool(row["is_summarized"]),
                    user_message_count=row["user_message_count"]
                )
                for row in rows
            ]


# Singleton instance
_storage: Optional[ChatStorage] = None


def get_chat_storage() -> ChatStorage:
    """Get the singleton ChatStorage instance."""
    global _storage
    if _storage is None:
        _storage = ChatStorage()
    return _storage
