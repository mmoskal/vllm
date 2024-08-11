#!/bin/bash

# ./run.sh -p 4,8,16 -l 32,64,128 -m "microsoft/Orca-2-13b" -c "true,false" --wandb

set -e

# Get the directory of the current script
DIR=$(dirname "$(readlink -f "$0")")

# Default grid search parameters
processes=(4)
prompt_lengths=(32)
models=("microsoft/Orca-2-13b")
use_constraint_options=(true)
wandb=false  # Default to not using wandb

# Parse command-line options to override defaults
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -p|--processes) IFS=',' read -ra processes <<< "$2"; shift ;;
    -l|--prompt_lengths) IFS=',' read -ra prompt_lengths <<< "$2"; shift ;;
    -m|--models) IFS=',' read -ra models <<< "$2"; shift ;;
    -c|--use_constraint) IFS=',' read -ra use_constraint_options <<< "$2"; shift ;;
    -w|--wandb) wandb=true; shift ;;  # Turn on wandb if the flag is set
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Deduplicate the options
processes=($(echo "${processes[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
prompt_lengths=($(echo "${prompt_lengths[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
use_constraint_options=($(echo "${use_constraint_options[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
models=($(echo "${models[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Print the grid search parameters
echo "Running grid search with the following parameters:"
echo "Processes: ${processes[@]}"
echo "Prompt lengths: ${prompt_lengths[@]}"
echo "Models: ${models[@]}"
echo "Use constraint: ${use_constraint_options[@]}"

# Grid search loop
for proc in "${processes[@]}"; do
  for prompt_len in "${prompt_lengths[@]}"; do
    for model in "${models[@]}"; do
      for use_constraint in "${use_constraint_options[@]}"; do
        # Construct the command
        cmd="python $DIR/create_delete_multi.py --port 4242 --processes $proc --prompt_len $prompt_len --model \"$model\""

        # Add optional flags
        [ "$use_constraint" = true ] && cmd+=" --use_constraint"
        [ "$wandb" = true ] && cmd+=" --wandb"

        # Run the command
        echo "Running command: $cmd"
        eval $cmd
      done
    done
  done
done