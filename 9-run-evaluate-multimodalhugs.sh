export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# split2_restored ca (es->ca)
config_path="configs/config_ca.yaml"
checkpoint_path="" #Path to best checkpoint after training.
results_path="final_results/"
split_path="metadata/"

sbatch scripts/run_evaluate_multimodal_hugs.sh $config_path \
$split_path ca $results_path $checkpoint_path sacrebleu 

sbatch scripts/run_evaluate_multimodal_hugs.sh $config_path \
$split_path ca $results_path $checkpoint_path chrf  

sbatch scripts/run_evaluate_other_metrics.sh -f $results_path/sacrebleu/predictions_labels.txt \
-o $results_path -m bertscore comet labse blaser -s cat -t cat
