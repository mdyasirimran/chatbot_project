import sqlite3
import datetime
import uuid

DATABASE_NAME = 'chatbot.db'

def create_connection():
    """Create a database connection to the SQLite database specified by DATABASE_NAME"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_tables():
    """Create tables if they do not exist"""
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
            """)
            conn.commit()
            print("Tables created successfully or already exist.")
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
        finally:
            conn.close()

def get_or_create_user(session_id):
    """Get user ID by session_id, or create a new user if not found"""
    conn = create_connection()
    user_id = None
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE session_id = ?", (session_id,))
            user = cursor.fetchone()
            if user:
                user_id = user[0]
            else:
                timestamp = datetime.datetime.now().isoformat()
                cursor.execute("INSERT INTO users (session_id, created_at) VALUES (?, ?)", (session_id, timestamp))
                conn.commit()
                user_id = cursor.lastrowid
                print(f"New user created with session_id: {session_id}, user_id: {user_id}")
        except sqlite3.Error as e:
            print(f"Error getting/creating user: {e}")
        finally:
            conn.close()
    return user_id

def log_conversation(session_id, user_message, bot_response):
    """Log a user's message and the bot's response"""
    conn = create_connection()
    if conn:
        try:
            user_id = get_or_create_user(session_id)
            if user_id:
                timestamp = datetime.datetime.now().isoformat()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (user_id, user_message, bot_response, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, user_message, bot_response, timestamp)
                )
                conn.commit()
                print(f"Conversation logged for user {session_id}")
            else:
                print(f"Could not log conversation: User with session_id {session_id} not found or created.")
        except sqlite3.Error as e:
            print(f"Error logging conversation: {e}")
        finally:
            conn.close()

def get_conversation_history(session_id, limit=5):
    """Retrieve conversation history for a given session_id"""
    conn = create_connection()
    history = []
    if conn:
        try:
            user_id = get_or_create_user(session_id)
            if user_id:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_message, bot_response, timestamp FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit)
                )
                history = cursor.fetchall()
                history.reverse() # Show in chronological order
        except sqlite3.Error as e:
            print(f"Error retrieving conversation history: {e}")
        finally:
            conn.close()
    return history

if __name__ == "__main__":
    create_tables()
    # Example usage:
    # session_id_1 = str(uuid.uuid4())
    # log_conversation(session_id_1, "Hello chatbot", "Hi there!")
    # log_conversation(session_id_1, "How are you?", "I'm doing great, thanks for asking!")
    # history = get_conversation_history(session_id_1)
    # print(f"\nConversation History for {session_id_1}:")
    # for msg in history:
    #     print(f"User: {msg[0]} | Bot: {msg[1]} | Time: {msg[2]}")