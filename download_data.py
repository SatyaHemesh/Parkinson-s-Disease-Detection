import os
import urllib.request

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
directory = "data"
filename = "parkinsons.data"
filepath = os.path.join(directory, filename)

if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Created directory: {directory}")

print(f"Downloading dataset from {url}...")
try:
    urllib.request.urlretrieve(url, filepath)
    print(f"✅ Success! Dataset saved to: {filepath}")
except Exception as e:
    print(f"❌ Error downloading file: {e}")