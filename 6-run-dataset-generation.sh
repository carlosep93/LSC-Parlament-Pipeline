#!/bin/bash

source load_ffmpeg.sh
source env3.10/bin/activate

search_dir='parlament/yamls/'
for session in "$search_dir"/*_speaker.yaml
do
  session=${session/$search_dir/}
  session=${session%%_*}
  session=${session:1}
  echo $session 
  python src/clusters_launcher.py \
    --process dataset_generation \
    --data-directory parlament/ \
    --ca-nemo-model nemo_models/stt_ca_conformer_ctc_large.nemo  \
    --es-nemo-model nemo_models/stt_es_conformer_ctc_large.nemo \
    --session-id $session \
    --sbatch   
done 

