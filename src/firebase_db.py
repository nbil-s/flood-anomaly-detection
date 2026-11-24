import os
import json

import firebase_admin
from firebase_admin import credentials, firestore as admin_firestore
from google.cloud import firestore as google_firestore


def get_firestore_client():
    if not firebase_admin._apps:
        service_account_json = os.environ.get(
            "FIREBASE_SERVICE_ACCOUNT_JSON"
        )

        if service_account_json:
            service_account_info = json.loads(service_account_json)
            credential = credentials.Certificate(service_account_info)

        else:
            key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            if not key_path or not os.path.isfile(key_path):
                raise FileNotFoundError(
                    "Set FIREBASE_SERVICE_ACCOUNT_JSON or "
                    "GOOGLE_APPLICATION_CREDENTIALS."
                )

            credential = credentials.Certificate(key_path)

        firebase_admin.initialize_app(credential)

    return admin_firestore.client()


db = get_firestore_client()


def save_sensor_reading(reading, detection=None):
    document = dict(reading)
    document["received_at"] = google_firestore.SERVER_TIMESTAMP

    if detection is not None:
        document["detection"] = dict(detection)

    _, doc_ref = db.collection("sensor_readings").add(document)
    return doc_ref