
import os
import firebase_admin
from firebase_admin import credentials, firestore

def get_firestore_client():
    if not firebase_admin._apps:
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not key_path or not os.path.isfile(key_path):
            raise FileNotFoundError(
                "Set GOOGLE_APPLICATION_CREDENTIALS to your Firebase key."
            )

        credential = credentials.Certificate(key_path)
        firebase_admin.initialize_app(credential)

    return firestore.client()


db = get_firestore_client()


def save_sensor_reading(reading, detection=None):
    """Save a sensor reading and return its Firestore document reference."""
    document = dict(reading)

    if detection is not None:
        document["detection"] = dict(detection)

    _, doc_ref = db.collection("sensor_readings").add(document)
    return doc_ref

