import json
import os
from pathlib import Path


def update_jsons_video_paths(parlament_data_dir, json_dir) -> None:
    """
    Iterates through all video files in the video_dir and updates the JSON files
    to include the correct 'path_audio' for each video.
    """
    video_dir = parlament_data_dir / "videos"
    audio_mapping_post2019 = {}

    # Iterate over all video files in the directory
    for video_filename in os.listdir(video_dir):
        if video_filename.endswith(".mp4"):
            video_dir / video_filename

            # Extract id and index from the filename (assuming the format is id_index.mp4)
            id_part, index_part = video_filename.rsplit("_", 1)
            video_url_placeholder = (
                f"{id_part}_{index_part.split('.')[0]}"  # e.g., "1234_01"
            )
            print(id_part, index_part, video_url_placeholder)

            # Map the placeholder URL to the local video path
            audio_mapping_post2019[
                video_url_placeholder
            ] = f"parlament/audios/{video_url_placeholder}.wav"

            # Update JSON files in the specified directory
            for json_filename in os.listdir(json_dir):
                if json_filename.endswith(".json"):
                    local_json_filepath = json_dir / json_filename
                    with open(local_json_filepath, "r") as json_file:
                        r = json.load(json_file)

                    json_id_part, json_index_part = json_filename.rsplit("_", 1)
                    json_url_placeholder = f"{json_id_part}_{json_index_part.split('.')[0]}"  # e.g., "1234_01"
                    print(json_id_part, json_index_part, json_url_placeholder)
                    # Update the 'path_audio' key in the JSON data
                    updated = False
                    for session in r.get("data", {}).get("sessions", []):
                        # Assuming the placeholder matches with the correct video
                        if json_id_part == id_part:
                            session["path_audio"] = str(
                                audio_mapping_post2019[video_url_placeholder]
                            )
                            updated = True

                    # Save changes back to the JSON file if updated
                    if updated:
                        with open(local_json_filepath, "w") as json_file:
                            json.dump(r, json_file, indent=2)
                        print(
                            f"[INFO]: Updated {local_json_filepath} with new path_audio"
                        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python update_videos.py <parlament_data_dir> <json_dir>")
        sys.exit(1)

    parlament_data_dir = Path(sys.argv[1])
    json_dir = Path(sys.argv[2])
    print("[INFO]: Updating jsons")
    update_jsons_video_paths(parlament_data_dir, json_dir)
    print("[INFO]: Jsons updated")
