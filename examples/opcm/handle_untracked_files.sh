#!/bin/bash

# Script to handle untracked files that are preventing git checkout

echo "Handling untracked files that are preventing git checkout..."

# Option 1: Create a backup of the untracked files
mkdir -p backup_untracked_files
echo "Creating backup of untracked files in ./backup_untracked_files/"

# Copy the files to backup
cp -r config/method/DOGE_TA backup_untracked_files/config_method_DOGE_TA
cp -r fusion_bench/method/DOGE_TA backup_untracked_files/fusion_bench_method_DOGE_TA

# Option 2: Remove the untracked files
echo "Do you want to remove the untracked files? (y/n)"
read answer

if [ "$answer" = "y" ]; then
    echo "Removing untracked files..."
    rm -rf config/method/DOGE_TA
    rm -rf fusion_bench/method/DOGE_TA
    echo "Files removed. You can now run 'git checkout main'"
else
    echo "Files were not removed, but backups were created."
    echo "You can manually remove the files or use one of these Git commands:"
    echo "  - git stash -u (to stash the changes temporarily)"
    echo "  - git add . && git commit -m \"Add DOGE_TA files\" (to commit the files)"
    echo "  - git clean -f (to forcefully remove untracked files - use with caution)"
fi

echo "Done!" 