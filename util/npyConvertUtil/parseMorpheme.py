import json

def parse_morpheme(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    morphemes = []
    for segment in data.get("data", []):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        attributes = segment.get("attributes", [])
        labels = [attr.get("name", "") for attr in attributes]
        label_str = " ".join(labels)
        morphemes.append({"start": start, "end": end, "label": label_str})

    return morphemes