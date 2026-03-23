from formwave_ai.storage.gcs_client import GCSClient
from formwave_ai.storage.paths import GCSPaths

BUCKET = "formwave-data"

client = GCSClient(BUCKET)

video_id = "video_001"

client.upload_file(
    "formwave_ai/storage/sample.txt",
    f"{GCSPaths.POSES}{video_id}.txt"
)
