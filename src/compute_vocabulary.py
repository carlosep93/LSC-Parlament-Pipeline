import os
import tqdm
import pandas as pd
from collections import Counter

metadata_path = "metadata/raw_data/"
audios_path = "segments/"

language = "ca"

results = dict()

for dataset in ["train", "validation", "test"]:
    # read tsv file
    df = pd.read_table(os.path.join(metadata_path, language, f"{dataset}.tsv"))

    results[dataset] = Counter()

    tmp_folder = "tmp"

    os.makedirs(tmp_folder, exist_ok=True)

    # get audio files
    for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        results[dataset].update(row.output.split(" "))

results["total"] = results["train"] + results["validation"] + results["test"]

train_words = set(results["train"].keys())
test_words = set(results["test"].keys())

not_in_train = [word for word in test_words if word not in train_words]

# this tells us the frequency of each count
not_in_train_counts = Counter(results["test"][key] for key in not_in_train)

# save results
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print(f"Mean WER: {df.WER.mean()}")
