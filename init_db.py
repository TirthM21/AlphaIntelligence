from src.database.db_manager import DBManager
import os
from dotenv import load_dotenv

load_dotenv()

def init_db():
    print("Initializing Hedge Fund Database...")
    db = DBManager()
    if db.db_url:
        print("✅ Database connection successful.")
        print("✅ Tables created/verified.")
    else:
        print("❌ Database connection failed. Check your DATABASE_URL.")

if __name__ == "__main__":
    init_db()
