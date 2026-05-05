import pandas as pd
import os
from pydub import AudioSegment
import whisper
import evaluate
import tqdm
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

metadata_path = "metadata/"
audios_path = "segments/"

language = "ca"

model = whisper.load_model("large")

results = dict()

for dataset in ["train", "validation", "test"]:

    results[dataset] = []

    # read tsv file
    df = pd.read_table(os.path.join(metadata_path, language, f"{dataset}.tsv"))

    tmp_folder = "tmp"

    os.makedirs(tmp_folder, exist_ok=True)

    save_segments = False

    wer = evaluate.load("wer")

    # get audio files
    for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        session = row.signal.split("/")[-1].split("_")[0]
        audio_path = os.path.join(audios_path, session, row.signal.split("/")[-1].replace(".mp4", ".wav"))

        audio_start_ms = row.audio_start
        audio_end_ms = row.audio_end

        audio = AudioSegment.from_wav(audio_path)
        clip = audio[audio_start_ms:audio_end_ms]

        # compute ASR with whisper
        # Export temporary file
        temp_path = os.path.join(tmp_folder, "temp_segment.wav")
        clip.export(temp_path, format="wav")

        out = model.transcribe(temp_path, language="ca", fp16=False)

        if save_segments:
            # get POSE video
            with open(row.signal, "rb") as f:
                pose = Pose.read(f.read())

            fps = pose.body.fps
            pose_start_ms = row.signal_start
            pose_end_ms = row.signal_end
            start_frame = int(pose_start_ms / 1000 * fps)
            end_frame = int(pose_end_ms / 1000 * fps)
            cropped_body = pose.body[start_frame:end_frame]
            cropped_pose = Pose(header=pose.header, body=cropped_body)

            with open(os.path.join(tmp_folder, "pose_segment.pose"), "wb") as data_buffer:
                cropped_pose.write(data_buffer)

            v = PoseVisualizer(cropped_pose)

            v.save_video(os.path.join(tmp_folder, "pose_segment.mp4"), v.draw())

        # compute WER
        target = row.output
        transcribed = out["text"].strip()

        results[dataset].append({
            "target": target,
            "transcription": transcribed,
            "WER": wer.compute(references=[target], predictions=[transcribed])
        })

rows = [
    {
        "dataset": dataset,
        "target": item["target"],
        "transcription": item["transcription"],
        "WER": item["WER"],
    }
    for dataset, items in results.items()
    for item in items
]

# save results
df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)

print(f"Mean WER: {df.WER.mean()}")
