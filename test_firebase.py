
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone

key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

if not key_path or not os.path.isfile(key_path):
    raise FileNotFoundError(
        "Firebase key not found. Set GOOGLE_APPLICATION_CREDENTIALS first."
    )

if not firebase_admin._apps:
    credential = credentials.Certificate(key_path)
    firebase_admin.initialize_app(credential)

db = firestore.client()

test_data = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "message": "Firebase connection test successful",
    "test_record": True
}

doc_ref = db.collection("sensor_readings").add(test_data)

print("Firebase connection successful.")
print("Test document ID:", doc_ref[1].id)
