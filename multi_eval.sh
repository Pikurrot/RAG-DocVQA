#!/bin/bash

CONFIG="configs/multi_eval.yml"

# 1) Parse arrays using helper python script (same as before)
PAGE_RET_ARRAY=($(python src/parse_yml.py "$CONFIG" data page_retrieval))
GPU_ARRAY=($(python src/parse_yml.py "$CONFIG" runtime visible_devices))

# Function to see if a GPU is busy
# We assume that if there's any screen whose name includes "_gpuX_eval" then GPU X is busy
function is_gpu_busy() {
    local gpu_id=$1
    # check if screen with "_gpu<gpu_id>_eval" is running
    screen -ls | grep -q "_gpu${gpu_id}_eval"
    return $?  # grep returns 0 if found (meaning busy)
}

i=0
for pr in "${PAGE_RET_ARRAY[@]}"; do
  
  while true; do
    gpu_index=$((i % ${#GPU_ARRAY[@]}))
    gpu="${GPU_ARRAY[$gpu_index]}"

    # Is GPU 'gpu' busy?
    if ! is_gpu_busy "$gpu"; then
      # Not busy, so launch
      screen_name="${pr}_gpu${gpu}_eval"
      echo "Launching page_retrieval=$pr on GPU $gpu (screen $screen_name)"
      screen -L -dmS "$screen_name" python eval.py "$CONFIG" page_retrieval="$pr" visible_devices="$gpu"
      # Move to next GPU for next job
      ((i++))
      # Break out of the while loop to move to next retrieval
      break
    else
      # GPU is busy, wait a bit and try again
      sleep 5
    fi
  done

done

# Optionally wait until *all* screens finish
# This will block your shell until there are no "_eval" screens left
# (not strictly required if you're okay to let them run in the background)
while true; do
  RUNNING_SCREENS=$(screen -ls | grep "_eval" | wc -l)
  if [ "$RUNNING_SCREENS" -eq 0 ]; then
    echo "All jobs finished!"
    break
  fi
  echo "$RUNNING_SCREENS jobs still running..."
  sleep 10
done
