#!/usr/bin/env bash
CONFIG="configs/multi_eval.yml"

# Use mapfile to store each line of Python output as one array element:
mapfile -t LINES < <(python src/flatten_multi_yml.py "$CONFIG")
# Now:
#   LINES[0] = "5 6 7 8 9"
#   LINES[1] = "retrieval.chunk_num retrieval.rerank_filter_tresh retrieval.rerank_max_chunk_num runtime.save_name_append"
#   LINES[2] = "10"
#   LINES[3] = "10 0 1 1001"
#   ...
# etc.

#########################################
# 1) The first line is the GPU list or NO_VISIBLE_DEVICES
#########################################
line1="${LINES[0]}"
if [ "$line1" == "NO_VISIBLE_DEVICES" ]; then
  GPU_ARRAY=()
  echo "No GPU array found."
else
  # Split line1 by spaces to get the GPU IDs
  IFS=' ' read -ra GPU_ARRAY <<< "$line1"
fi

#########################################
# 2) The second line: either multi-run keys or NO_MULTI_PARAMS
#########################################
line2="${LINES[1]}"
if [ "$line2" == "NO_MULTI_PARAMS" ]; then
  echo "No multi-run parameters found in $CONFIG."
  exit 0
fi

# Otherwise, read them into an array:
IFS=' ' read -ra MULTI_KEYS <<< "$line2"

#########################################
# 3) The third line is the integer N
#########################################
N="${LINES[2]}"

#########################################
# 4) The next N lines contain parameter values
#########################################
ROWS=()
start_index=3
for (( i=0; i<N; i++ )); do
  ROWS[i]="${LINES[$((start_index + i))]}"
done

#########################################
# 5) A function to check if a GPU is busy
#########################################
function is_gpu_busy() {
    local gpu_id="$1"
    screen -ls | grep -q "_gpu${gpu_id}_eval"
    return $?
}

gpu_queue_index=0

#########################################
# 6) Launch each row in a screen
#########################################
for (( i=0; i<N; i++ )); do
  row="${ROWS[$i]}"
  # row might look like: "10 0 1 1001" (4 values) if we have 4 multi-keys
  IFS=' ' read -ra rowVals <<< "$row"

  # Build key=value overrides
  override_args=""
  for (( k=0; k<${#MULTI_KEYS[@]}; k++ )); do
    key="${MULTI_KEYS[$k]}"
    val="${rowVals[$k]}"
    override_args="$override_args $key=$val"
  done

  #########################################
  # 7) Pick a GPU in round-robin (optional)
  #########################################
  if [ ${#GPU_ARRAY[@]} -eq 0 ]; then
    # No GPUs => just spawn in screen
    screen_name="combo${i}_eval"
    echo "Launching row $i with overrides [$override_args] (no GPU assigned)."
    screen -L -dmS "$screen_name" \
      python eval.py "$CONFIG" $override_args
  else
    while true; do
      gpu_index=$((gpu_queue_index % ${#GPU_ARRAY[@]}))
      gpu="${GPU_ARRAY[$gpu_index]}"

      if ! is_gpu_busy "$gpu"; then
        screen_name="combo${i}_gpu${gpu}_eval"
        echo "Launching row $i with overrides [$override_args] on GPU $gpu."

        # shellcheck disable=SC2086
        screen -L -dmS "$screen_name" \
          python eval.py "$CONFIG" visible_devices="$gpu" $override_args

        ((gpu_queue_index++))
        break
      else
        sleep 5
      fi
    done
  fi
done

#########################################
# 8) (Optional) Wait for all screens
#########################################
while true; do
  RUNNING_SCREENS=$(screen -ls | grep "_eval" | wc -l)
  if [ "$RUNNING_SCREENS" -eq 0 ]; then
    echo "All jobs finished!"
    break
  fi
  echo "$RUNNING_SCREENS still running..."
  sleep 10
done
