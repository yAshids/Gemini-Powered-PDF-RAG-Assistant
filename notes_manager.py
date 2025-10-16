import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from bson.objectid import ObjectId
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "ai_hub")
NOTES_COL = os.getenv("NOTES_COL", "notes")

def _client():
    c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    try:
        c.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        raise RuntimeError(f"MongoDB not reachable at {MONGODB_URI}. Start mongod or update MONGODB_URI.") from e
    return c

_client_singleton = None
def _db():
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = _client()
    return _client_singleton[MONGODB_DB]

collection = _db()[NOTES_COL]

def save_note(title, content):
    doc = {"title": title, "content": content, "updated_at": datetime.utcnow()}
    return collection.insert_one(doc).inserted_id

def get_all_notes():
    return list(collection.find({}, {"title":1, "content":1}).sort([("_id", -1)]))

def delete_note(note_id):
    return collection.delete_one({"_id": ObjectId(note_id)})

def update_note(note_id, title, content):
    return collection.update_one({"_id": ObjectId(note_id)}, {"$set": {"title": title, "content": content, "updated_at": datetime.utcnow()}})

def delete_all_notes():
    res = collection.delete_many({})  # delete all docs safely
    return res.deleted_count
