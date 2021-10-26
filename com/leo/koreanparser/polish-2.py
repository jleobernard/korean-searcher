import os
import re
from pathlib import Path

import pandas as pd

annotation_files = [
    f"/opt/data/korean-subs/store/{f}/{f}.csv" for f in ["okay-not-okay-03", "okay-not-okay-ep02-0", "so-not-woth-it-01", "so-not-woth-it-02", "sweet-home-01", "uncanny-counter-02", "uncanny-counter-03"]]
for annotation_file_name in annotation_files:
    prefix = os.path.splitext(os.path.basename(annotation_file_name))[0]
    directory = Path(annotation_file_name).resolve().parent
    df_in = pd.read_csv(annotation_file_name)

    def clean_subs(annotation):
        subs = annotation["subs"]
        subs = re.sub("\[.+?]", "", subs)
        subs = re.sub("\[.+?", "", subs)
        subs = re.sub("\(.+?\)", "", subs)
        subs = re.sub("\(.+?", "", subs)
        data = {}
        data["subs"] = subs.strip()
        data["start"] = annotation["start"]
        data["end"] = annotation["end"]
        i = 0
        j = 0
        look_for = None
        while f"parsed_{i}" in annotation:
            value = annotation[f"parsed_{i}"]
            if look_for is None:
                if "(" == value:
                    look_for = ")"
                elif "[" == value:
                    look_for = "]"
                elif not pd.isna(value):
                    data[f"parsed_{j}"] = value
                    data[f"parsed_{j + 1}"] = annotation[f"parsed_{i + 1}"]
                    j += 2
            elif value == look_for:
                look_for = None
            i += 2
        for h in range(j, i):
            data[f"parsed_{h}"] = ""
        return data

    data = []
    curr_annotation = None
    curr_subs = None
    curr_frame_start = None
    curr_frame_end = None

    for i, annotation in df_in.iterrows():
        annotation = clean_subs(annotation)
        subs = annotation['subs']
        if subs is not None and not subs == "":
            data.append(annotation)
    df = pd.DataFrame(data=data)
    df.to_csv(f"{directory}/{prefix}.csv", index=False)

