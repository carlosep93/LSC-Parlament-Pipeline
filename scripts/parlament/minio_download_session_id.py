import json
import os
from argparse import ArgumentParser
from pathlib import Path

import requests
from dotenv import dotenv_values
from minio import Minio

config = dotenv_values(".env")

# Not using MinIO currently
MINIO_ENDPOINT = config["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = config["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = config["MINIO_SECRET_KEY"]
MINIO_BUCKET_NAME = config["MINIO_BUCKET_NAME"]

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


def load_processed_sessions(processed_sessions_file):
    """
    Returns the list of already processed sessions.
    """

    if os.path.exists(processed_sessions_file):
        with open(processed_sessions_file, "r") as file:
            return json.load(file)
    else:
        return []


def save_processed_sessions(processed_sessions, processed_sessions_file):
    """
    Saves the session that have already been processed.
    """

    with open(processed_sessions_file, "w") as file:
        json.dump(processed_sessions, file)


def get_jsons(id, json_dir):
    """
    Prameters: json id, and json storage directory.
    Downloads the json through an API and saves it.
    Returns the PDF's and Videos URLS.
    """

    url_detall = "https://www.parlament.cat/api/aina/v1"
    offset = 1
    step = 1
    pdfs, video_urls = [], []

    while True:
        url = f"{url_detall}/sessions_detall?format=json&sessio_id={id}&page_offset={offset}&page_limit=10"
        print(url)

        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON responsex
        data = response.json()

        if len(data.get("data", {}).get("sessions", [])) == 0:
            break

        sessions = data["data"]["sessions"]
        for session in sessions:
            video_url = session.get("video_qmitja_mp4_url")
            pdf_url = session.get("sessio_dspc_url")
            if video_url.endswith(".mp4") and pdf_url.endswith(".pdf"):
                pdfs.append(pdf_url)
                video_urls.append(video_url)
                with open(json_dir / f"{id}_{step:02d}.json", "w") as outfile:
                    json.dump(data, outfile, indent=2)

        offset += 10
        step += 1

    if not pdfs or not video_urls:
        return None

    unique_pdf_urls = list(dict.fromkeys(pdfs))
    unique_video_urls = list(dict.fromkeys(video_urls))

    return unique_pdf_urls, unique_video_urls


def download_pdfs(parlament_data_dir, id, unique_pdf_urls) -> None:
    """
    Given PDF ID and URL, downloads it from aina API server.
    It saves the pdf to the indicated directory.
    """

    pdf_dir = parlament_data_dir / "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    for i, pdf_url in enumerate(unique_pdf_urls):
        print(f"Downloading PDF {i+1} from: {pdf_url}")
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_filename = f"{id}_{i+1:02d}.pdf"

            local_pdf_filename = pdf_dir / pdf_filename
            with open(local_pdf_filename, "wb") as outfile:
                outfile.write(response.content)
        else:
            print(f"[ERROR]: Failed to fetch PDF from URL: {pdf_url}")


def download_videos(parlament_data_dir, id, unique_video_urls, json_dir) -> None:
    """
    Given Video ID and URL, downloads it from aina API server.
    It saves the video to the indicated directory.
    """
    video_dir = parlament_data_dir / "videos"
    audio_mapping_post2019 = {}

    for i, video_url in enumerate(unique_video_urls):
        print(f"[INFO]: Downloading Video {i+1} from: {video_url}")

        response = requests.get(video_url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            video_filename = f"{id}_{i+1:02d}.mp4"
            local_video_filename = video_dir / video_filename

            audio_mapping_post2019[video_url] = local_video_filename

            # Open a file to write the video data
            with open(local_video_filename, "wb") as file:
                # Write the data in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(
                f"[INFO]: Video downloaded successfully and saved as {local_video_filename}"
            )

        else:
            print(
                f"[ERROR]: Failed to download video. Status code: {response.status_code}"
            )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-directory",
        "-d",
        help="The relative path to the pipeline data directory",
        default="data/parlament",
    )

    args = parser.parse_args()
    parlament_data_dir = Path(args.data_directory)

    # check sessions already downloaded
    json_dir = parlament_data_dir / "jsons"
    processed_sessions_file = parlament_data_dir / "processed_sessions.json"
    processed_sessions = load_processed_sessions(processed_sessions_file)

    session_ids = [21504702]

    # Download JSON, PDF and VIDEO
    for session_id in session_ids:
        if session_id in processed_sessions:
            print(f"Session {session_id} has already been processed. Skipping.")
        else:
            print(f"Processing session with ID: {session_id}")
            result = get_jsons(session_id, json_dir)
            if result is not None:
                pdfs, videos_urls = result
                print(f"Session id {session_id} urls are \n {pdfs} \n {videos_urls}")
                download_pdfs(parlament_data_dir, session_id, pdfs)
                download_videos(parlament_data_dir, session_id, videos_urls, json_dir)

            else:
                print(
                    f"[WARNING]: Discarding session {session_id} due to lack of audio or pdfs"
                )

    processed_sessions = parlament_data_dir / "processed_sessions.json"
    save_processed_sessions(session_ids, processed_sessions)


if __name__ == "__main__":
    main()
