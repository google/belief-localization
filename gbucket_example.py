from google.cloud import storage


def gcs_test(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket('research-brain-belief-localization-xgcp')
  blob = bucket.blob('output/dummy.txt')
  blob.upload_from_string('Hi. This is a test.')

  print('Successfully created file: research-brain-belief-localization-xgcp/output/dummy.txt')

gcs_test()