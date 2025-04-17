#!/bin/bash

# Define the dataset, model, and the press names
dataset="wtq_tableqa_rag"
model="Qwen/Qwen2.5-7B-Instruct"
press_names=("finch")

# Define the data directories and their corresponding max capacity context values
data_dirs=("128" "256" "512" "1024" "2048")
# Mapping: if data_dir==256 then max_capacity_context==512, if data_dir==512 then max_capacity_context==1024, etc.
max_capacity_contexts=("256" "512" "1024" "2048" "4096")

# Check if the number of press names is less than or equal to the number of available GPUs
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ ${#press_names[@]} -gt $num_gpus ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
  exit 1
fi

# Iterate over press names
for press_index in "${!press_names[@]}"; do
  press="${press_names[$press_index]}"
  gpu="cuda:$press_index"  # assign a GPU to this press name based on its index

  # Sequentially iterate over each data directory and its corresponding max_capacity_context
  for i in "${!data_dirs[@]}"; do
    data_dir="${data_dirs[$i]}"
    max_capacity_context="${max_capacity_contexts[$i]}"
    
    echo "Running experiment for press_name: $press, data_dir: $data_dir with max_capacity_context: $max_capacity_context on GPU $gpu"
    
    # Run the experiment
    python evaluate_script.py --dataset "$dataset" --data_dir "$data_dir" --model "$model" --press_name "$press" --max_capacity_context "$max_capacity_context" --device "cuda:1"
  done
done

echo "All evaluations completed."
