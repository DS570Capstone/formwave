from google.cloud import storage
import os

class GCSClient:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path: str, gcs_path: str):
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded: {local_path} → gs://{self.bucket.name}/{gcs_path}")

    def download_file(self, gcs_path: str, local_path: str):
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"Downloaded: gs://{self.bucket.name}/{gcs_path} → {local_path}")

    def list_files(self, prefix: str):
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        return [blob.name for blob in blobs]

    def exists(self, gcs_path: str):
        blob = self.bucket.blob(gcs_path)
        return blob.exists()