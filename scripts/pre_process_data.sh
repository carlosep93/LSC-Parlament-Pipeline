#!/bin/bash

# AUDIO EXTRACTION
VIDEO_DIR="data/parlament/videos"
# Directory to save extracted audio files
AUDIO_DIR="data/parlament/audios"

# Create the audio directory if it does not exist
mkdir -p "$AUDIO_DIR"

# Loop through all video files in the video directory
for video_file in "$VIDEO_DIR"/*; do
    # Get the base name of the video file (without extension)
    base_name=$(basename "$video_file" | sed 's/\.[^.]*$//')
    # Define the output audio file path
    audio_file="$AUDIO_DIR/${base_name}.wav"

    # Extract audio using ffmpeg
    ffmpeg -i "$video_file" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$audio_file"

    echo "Extracted audio from $video_file to $audio_file"
done

echo "All audio files have been extracted."


# PERSON FRAME EXTRACTION

input_folder="data/parlament/videos"
output_folder="data/parlament/person_frame"

# Comprovar si la carpeta de sortida existeix, si no, crear-la
mkdir -p "$output_folder"

# Loop per processar cada fitxer de vídeo a la carpeta d'entrada
for input_file in "$input_folder"/*.mp4; do
  # Obtenir el nom del fitxer sense l'extensió
  filename=$(basename "$input_file" .mp4)
  
  # Definir el fitxer de sortida
  output_file="$output_folder/${filename}_frame.mp4"
  
  # Executar la comanda d'FFmpeg
  ffmpeg -i "$input_file" -vf "crop=330:330:30:75" -s 224x224 "$output_file"
done

# UPDATE JSONS

# Define the directories
PARLAMENT_DATA_DIR="data/parlament"
JSON_DIR="data/parlament/jsons"

# Run the Python script
python scripts/parlament/update_jsons.py "$PARLAMENT_DATA_DIR" "$JSON_DIR"

