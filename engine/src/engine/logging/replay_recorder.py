import json
import os
from datetime import datetime
from google.cloud import storage

def is_running_in_cloud():
    # Works for Cloud Run
    if os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN"):
        return True
    # Works for App Engine
    if os.getenv("GAE_ENV"):
        return True
    # Works for generic GCP detection
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        return True
    return False


class ReplayRecorder:
    def __init__(self, session_id, save_dir="replays"):
        self.session_id = session_id
        self.save_dir = save_dir
        self.data = {
            "session_id": session_id,
            "start_time": datetime.utcnow().isoformat(),
            "config": {},
            "agents": {},  # {agent_id: initial_state}
            "intents": []  # [{tick, agent_id, action}]
        }
        os.makedirs(save_dir, exist_ok=True)

    def set_config(self, config_dict):
        self.data["config"] = config_dict

    def register_agent(self, agent_id, initial_state=None):
        self.data["agents"][agent_id] = {
            "initial_state": initial_state or {}
        }

    def unregister_agent(self, agent_id):
        if agent_id in self.data["agents"]:
            del self.data["agents"][agent_id]

    def log_intent(self, tick, agent_id, action):
        self.data["intents"].append({
            "tick": tick,
            "agent_id": agent_id,
            "action": action
        })

    def save(self):
        filename = f"{self.session_id}.json"
        print('[ReplayRecorder] Saving replay data to', filename)

        if is_running_in_cloud():
            print('[ReplayRecorder] Detected cloud environment, uploading to GCS...')
            # Upload to GCS
            bucket_name = os.getenv("REPLAY_BUCKET", "replay_files")
            blob_path = f"replays/{filename}"
            try:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.upload_from_string(
                    json.dumps(self.data, indent=2),
                    content_type="application/json"
                )
                print(f"[ReplayRecorder] Uploaded to GCS: {bucket_name}/{blob_path}")
            except Exception as e:
                print(f"[ReplayRecorder] Failed to upload to GCS: {e}")
        else:
            # Save locally
            os.makedirs(self.save_dir, exist_ok=True)
            file_path = os.path.join(self.save_dir, filename)
            with open(file_path, "w") as f:
                json.dump(self.data, f, indent=2)
            print(f"[ReplayRecorder] Saved locally to {file_path}")
