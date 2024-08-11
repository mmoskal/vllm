import os
import subprocess
from jinja2 import Environment, FileSystemLoader
from multiprocessing import Pool
import argparse
import tempfile
import shutil


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate and run scripts with customized N and max_tokens values.")
parser.add_argument('--n_values', type=int, nargs='+', required=True, help="List of N values")
parser.add_argument('--max_tokens_values', type=int, nargs='+', required=True, help="List of max_tokens values")
parser.add_argument('--aici_script_path', type=str, required=True, help="Path where the aici.sh script is located")
parser.add_argument('--template_file', type=str, default='aici_template.py.jinja', help="Jinja2 template file")
parser.add_argument('--wandb', action='store_true', help="Upload the data to wandb")
args = parser.parse_args()

# Create a temporary directory for generated scripts and logs
tmp_dir = tempfile.mkdtemp(prefix="aici_run_")

# Load the Jinja2 template
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template(args.template_file)

# Function to generate the script file name based on N and max_tokens
def generate_output_filename(N, max_tokens):
    return f"aici_script_N{N}_T{max_tokens}.py"

# Function to render the template and save the output script
def render_script(N, max_tokens, output_file):
    output_content = template.render(N=N, max_tokens=max_tokens)
    output_path = os.path.join(tmp_dir, output_file)
    with open(output_path, 'w') as f:
        f.write(output_content)
    return output_path

# Function to run the generated script
def run_script(script_file):
    log_file = f"{os.path.splitext(script_file)[0]}.log"
    output_file = f"{os.path.splitext(script_file)[0]}.json"
    aici_script_path = args.aici_script_path
    command = f"sh {aici_script_path} run {script_file} --output_path {output_file}"
    with open(log_file, 'w') as log:
        subprocess.run(command, shell=True, stdout=log, stderr=log)
    return log_file

# Generate the list of configurations
configs = []
for N in args.n_values:
    for max_tokens in args.max_tokens_values:
        output_file = generate_output_filename(N, max_tokens)
        configs.append({"N": N, "max_tokens": max_tokens, "output_file": output_file})

# Generate the scripts
script_files = []
for config in configs:
    script_files.append(render_script(config['N'], config['max_tokens'], config['output_file']))

# Use multiprocessing to run the scripts concurrently once to warm up.
with Pool(processes=len(script_files)) as pool:
    log_files = pool.map(run_script, script_files)

# Print the log files generated
print("All tasks completed. Logs are saved in the following files:")
for log_file in log_files:
    print(log_file)

# Print the location of the temporary directory
print(f"All generated scripts and logs are located in the temporary directory: {tmp_dir}")

# Transfer the data to wandb
if args.wandb:
    import wandb
    print("Uploading data to wandb...")
    wandb.init(project="aici", group="create_sequence_aici")
    for log_file in log_files:
        wandb.save(log_file)
    print("Data uploaded to wandb.")