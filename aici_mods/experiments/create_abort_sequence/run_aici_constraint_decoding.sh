#!/bin/bash

# Default values for the arguments
N_VALUES="16 32"
MAX_TOKENS_VALUES="16"
AICI_SCRIPT_PATH="/workspaces/aici/scripts/aici.sh"
TEMPLATE_FILE="aici_template.py.jinja"
WANDB=false

# Function to display usage instructions
usage() {
    # sh run_aici_constraint.sh -n "16 32 64" -m "16 32"
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n, --n_values            Space-separated list of n_values (default: $N_VALUES)"
    echo "  -m, --max_tokens_values   Max tokens values (default: $MAX_TOKENS_VALUES)"
    echo "  -s, --script_path         Path to the AICI script (default: $AICI_SCRIPT_PATH)"
    echo "  -t, --template_file       Template file (default: $TEMPLATE_FILE)"
    echo "  -w, --wandb               Enable wandb (default: $WANDB)"
    echo "  -h, --help                Display this help message"
    echo " "
    echo "Example: $0 -n '16 32 64' -m '16 32' -s /workspaces/aici/scripts/aici.sh -t aici_template.py.jinja -w"
    exit 1
}

# Parse input arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--n_values)
            N_VALUES="$2"
            shift 2
            ;;
        -m|--max_tokens_values)
            MAX_TOKENS_VALUES="$2"
            shift 2
            ;;
        -s|--script_path)
            AICI_SCRIPT_PATH="$2"
            shift 2
            ;;
        -t|--template_file)
            TEMPLATE_FILE="$2"
            shift 2
            ;;
        -w|--wandb)
            WANDB=true
            shift 1
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Construct the command
CMD="python run_aici_scripts.py --n_values $N_VALUES --max_tokens_values $MAX_TOKENS_VALUES --aici_script_path $AICI_SCRIPT_PATH --template_file $TEMPLATE_FILE"
if [ "$WANDB" = true ]; then
    CMD="$CMD --wandb"
fi

# Run the command
echo "Running command: $CMD"
$CMD