import json
import gzip
 
# with gzip.open("2024-01-01-0.json.gz", "rb") as f:
#     data = json.loads(f.read().decode("utf-8"))

filename = "2024-01-01-0.json.gz"
with gzip.open(filename) as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        try:
            data = json.loads(line)
        except:
            continue
        print({
            "_op_type": "update",
            "_id": data['id'],
            "doc": data,
            "doc_as_upsert": True
        })
