import os
from google.cloud import storage

def get_client():
    return storage.Client()

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a specific file from GCS to local disk.
    """
    print(f"⬇️ Downloading: gs://{bucket_name}/{source_blob_name} ...")
    try:
        client = get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"✅ Downloaded to {destination_file_name}")
        return True
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        raise e

def upload_file(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a local file to a specific GCS bucket.
    """
    print(f"⬆️ Uploading: {destination_blob_name} to bucket {bucket_name} ...")
    try:
        client = get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print("✅ Upload complete.")
        return True
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        raise e

def find_latest_csv_blobs(bucket_name, prefix_path):
    """
    Finds all .csv files in the most recent subdirectory of a given GCS prefix.
    """
    print(f"🔎 Searching for latest CSVs in gs://{bucket_name}/{prefix_path}...")
    client = get_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix_path, match_glob="**/*.csv")

    latest_dir = None
    csv_files_in_latest_dir = []

    # Group blobs by directory
    dirs = {}
    for blob in blobs:
        dir_name = os.path.dirname(blob.name)
        if dir_name not in dirs:
            dirs[dir_name] = []
        dirs[dir_name].append(blob.name)

    # Find the latest directory lexicographically
    if dirs:
        latest_dir = max(dirs.keys())
        csv_files_in_latest_dir = dirs[latest_dir]
        print(f"🎯 Found {len(csv_files_in_latest_dir)} CSV(s) in latest directory: {latest_dir}")
        return csv_files_in_latest_dir

    print("⚠️ No CSVs found in any directory under the prefix.")
    return []

def find_blobs_in_folder(bucket_name, folder_path):
    """
    Finds all .csv files within a specific folder (prefix).
    """
    print(f"🔎 Searching for CSVs in gs://{bucket_name}/{folder_path}...")
    client = get_client()
    # Ensure folder path ends with / if not empty, to treat as directory
    if folder_path and not folder_path.endswith('/'):
        folder_path += '/'
        
    blobs = client.list_blobs(bucket_name, prefix=folder_path, match_glob="**/*.csv")
    csv_files = [blob.name for blob in blobs]
    
    if csv_files:
        print(f"🎯 Found {len(csv_files)} CSV(s) in folder: {folder_path}")
    else:
        print(f"⚠️ No CSVs found in folder: {folder_path}")
        
    return csv_files

def find_all_jsonl_in_path(bucket_name, prefix_path):
    """
    Searches for all .jsonl files within a GCS path, including subdirectories.
    This is useful for finding all output parts of a Vertex AI Batch Prediction job.
    """
    print(f"🔎 Searching for all JSONL files in gs://{bucket_name}/{prefix_path}...")
    client = get_client()
    bucket = client.bucket(bucket_name)
    
    # The glob '**/*.jsonl' will find all files ending with .jsonl in any subdirectory
    # under the given prefix.
    glob_pattern = f"{prefix_path}**/*.jsonl"
    blobs = bucket.list_blobs(match_glob=glob_pattern)
    
    blob_names = [blob.name for blob in blobs]
    
    if blob_names:
        print(f"🎯 Found {len(blob_names)} prediction file(s).")
    else:
        print(f"⚠️ No .jsonl files found in the specified path with glob '{glob_pattern}'.")
        
    return blob_names
