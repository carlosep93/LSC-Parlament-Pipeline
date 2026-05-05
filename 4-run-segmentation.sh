#!/bin/bash



search_dir='parlament/yamls/'
for session in "$search_dir"/*_speaker.yaml
do
  session=${session/$search_dir/}
  session=${session%%_*}
  session=${session:1}
  echo $session 
  python src/clusters_launcher.py \
    --data-directory parlament/ \
    --process segmentation \
    --session-id $session \
    --sbatch   
done

