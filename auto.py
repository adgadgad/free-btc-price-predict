import os
import time
import subprocess

# Path to your Git repository
repo_path = '/workspaces/opensource_BTC-price-predictor/'

# Path to the file you want to track
file_path = os.path.join(repo_path, 'btc_price_data_alpha_vantage_ful.csv')

# Function to run git commands
def run_git_command(command):
    result = subprocess.run(command, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode(), result.stderr.decode()

# Function to check if there are any changes to the file
def has_changes():
    # Check if the file has changes in git
    diff, _ = run_git_command(['git', 'diff', '--quiet', '--', file_path])
    return diff != ''

# Function to commit and push changes
def commit_and_push():
    print("Changes detected, committing and pushing the file...")
    run_git_command(['git', 'add', file_path])
    run_git_command(['git', 'commit', '-m', 'Auto-commit: Update BTC price data'])
    run_git_command(['git', 'push', 'origin', 'main'])

# Main loop to check every 10 minutes
while True:
    if has_changes():
        commit_and_push()
    else:
        print("No changes detected.")
    
    # Wait for 10 minutes (600 seconds)
    time.sleep(600)