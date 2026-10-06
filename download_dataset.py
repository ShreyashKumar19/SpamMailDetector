import urllib.request
import zipfile
import os


url = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"

print("Downloading dataset...")

urllib.request.urlretrieve(url, "sms_spam_collection.zip")

print("Download complete.")

with zipfile.ZipFile("sms_spam_collection.zip", "r") as zip_file:
    zip_file.extractall()

os.remove("sms_spam_collection.zip")

print("Dataset is ready.")
