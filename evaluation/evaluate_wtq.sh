dataset="wtq_tableqa"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
max_capacity_contexts=(2048)
press_names=("finch")

# Check if the number of press names is less than or equal to the number of available GPUs
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ ${#press_names[@]} -gt $num_gpus ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
  exit 1
fi

# Iterate over press names and compression ratios
for i in "${!press_names[@]}"; do
  press="${press_names[$i]}"
  
  # Run each press_name on a different GPU in the background
  (
    for max_capacity_context in "${max_capacity_contexts[@]}"; do
      echo "Running press_name: $press with max_capacity_context: $max_capacity_context on GPU cuda:$i"
      python evaluate_script.py --dataset $dataset --model $model --press_name $press --max_capacity_context $max_capacity_context --device "cuda:1"
    done
  ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."
