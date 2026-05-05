#!/bin/bash

touch visited.txt

search_dir='parlament/yamls/'
for session in "$search_dir"/*_speaker.yaml
do
  session=${session/$search_dir/}
  session=${session%%_*}
  session=${session:1}
  if ! grep -q $session visited.txt; then 
    echo $session 
    python src/clusters_launcher.py \
    --data-directory parlament/ \
    --process alignment \
    --session-id $session \
    --sbatch   
  fi
done

