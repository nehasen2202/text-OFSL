import json

base_json  = "base.json"
val_json   = "val.json"
novel_json = "novel.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

base_data  = load_json(base_json)
val_data   = load_json(val_json)
novel_data = load_json(novel_json)

allc = {
    "label_names": base_data.get("label_names", []),
    "image_names": [],
    "image_labels": [],
    "image_data": []
}

def add_split(data):
    allc["image_names"].extend(data.get("image_names", []))
    allc["image_labels"].extend(data.get("image_labels", []))

    if "image_data" in data:
        allc["image_data"].extend(data["image_data"])
    else:
        allc["image_data"].extend([""] * len(data.get("image_names", [])))

add_split(base_data)
add_split(val_data)
add_split(novel_data)

with open("allc.json", "w") as f:
    json.dump(allc, f)

print("✅ allc.json created successfully!")
print("Total images:", len(allc["image_names"]))
print("Total labels:", len(allc["label_names"]))
print("Total captions:", len(allc["image_data"]))

