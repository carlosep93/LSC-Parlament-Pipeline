import json
import os
from argparse import ArgumentParser

from databases import MongoDatabase


def store_files(pipeline, files_path, collection_name):
    db = MongoDatabase(pipeline)
    subdirectories = [
        d for d in os.listdir(files_path) if os.path.isdir(os.path.join(files_path, d))
    ]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(files_path, subdirectory)
        print(f"Processing files in directory: {subdirectory_path}")
        json_files = [f for f in os.listdir(subdirectory_path) if f.endswith(".json")]

        for json_file in json_files:
            json_file_path = os.path.join(subdirectory_path, json_file)
            print(f"Processing JSON file: {json_file_path}")

            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

            db.insert(data)

    if collection_name:
        db.renameCollection(pipeline, collection_name)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--pipeline",
        "-p",
        help="Pipeline to use, either corts / parlament / ib3.",
        required=True,
    )

    parser.add_argument(
        "--files-path",
        "-f",
        help="Path where the final output JSON files are located.",
        required=True,
    )

    parser.add_argument(
        "--collection-name",
        "-c",
        help="Name of the collection in which data will be stored.",
        default=None,
    )

    args = parser.parse_args()

    pipeline = args.pipeline
    files_path = args.files_path
    collection_name = args.collection_name
    store_files(pipeline, files_path, collection_name)


if __name__ == "__main__":
    main()
