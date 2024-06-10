import requests
import json

API_TOKEN = "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"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

API_URL = "https://huggingface.co/api/datasets/andrewsleader/TGB/croissant" #"https://huggingface.co/api/datasets/ibm/duorc/croissant"


def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
print (data)

with open('tgb2_croissant.json', 'w') as f:
    json.dump(data, f)