
import os
from pymongo import MongoClient

try:
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client.get_database('GESTURE')
    col = db.get_collection('history')
except Exception as e:
    raise ImportError("MongoDB not connected")