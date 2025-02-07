#!/bin/bash
CONFIG="configs/multi_eval.yml"

PAGE_RET_ARRAY=($(python src/parse_yml.py "$CONFIG" data page_retrieval))
BS_ARRAY=($(python src/parse_yml.py "$CONFIG" data batch_size))
GPU_ARRAY=($(python src/parse_yml.py "$CONFIG" runtime visible_devices))

# Function to detect if a GPU is busy
function is_gpu_busy() {
    local gpu_id=$1
    screen -ls | grep -q "_gpu${gpu_id}_eval"
    return $?
}

gpu_queue_index=0
for i in "${!PAGE_RET_ARRAY[@]}"; do
  pr="${PAGE_RET_ARRAY[$i]}"
  bs="${BS_ARRAY[$i]}"

  while true; do
    gpu_index=$((gpu_queue_index % ${#GPU_ARRAY[@]}))
    gpu="${GPU_ARRAY[$gpu_index]}"

    if ! is_gpu_busy "$gpu"; then
      screen_name="${pr}_gpu${gpu}_eval"
      echo "Launching page_retrieval=$pr batch_size=$bs on GPU $gpu (screen $screen_name)"
      screen -L -dmS "$screen_name" \
        python eval.py "$CONFIG" page_retrieval="$pr" batch_size="$bs" visible_devices="$gpu"
      ((gpu_queue_index++))
      break
    else
      sleep 5
    fi
  done
done

# Optional final wait loop
while true; do
  RUNNING_SCREENS=$(screen -ls | grep "_eval" | wc -l)
  if [ "$RUNNING_SCREENS" -eq 0 ]; then
    echo "All jobs finished!"
    break
  fi
  echo "$RUNNING_SCREENS jobs still running..."
  sleep 10
done
