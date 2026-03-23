from google.cloud import storage

client = storage.Client()

print("Connected!")

for bucket in client.list_buckets():
    print(bucket.name)