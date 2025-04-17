#!/usr/bin/env python
"""
Dataset Version Management Utility

This script helps manage different versions of cleaned datasets for fine-tuning.
It allows you to:
1. Backup the current dataset to a versioned folder
2. List available dataset versions
3. Restore a specific dataset version for training
"""

import os
import glob
import shutil
import argparse
import re
import datetime

# Paths
DATA_DIR = "../data/processed"
BACKUP_DIR = "../backup"

def list_dataset_versions():
    """List all available dataset versions in the backup directory"""
    dataset_dirs = glob.glob(f"{BACKUP_DIR}/cleaned-dataset*")
    versions = [os.path.basename(d) for d in dataset_dirs]
    versions.sort()
    return versions

def backup_current_dataset(version_name=None):
    """Backup the current dataset to a new versioned folder"""
    # Check if there is a dataset to backup
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"No dataset found in {DATA_DIR}")
        return False
    
    # Determine version name
    if version_name is None:
        # Check existing versions to determine the next number
        versions = list_dataset_versions()
        version_numbers = [int(re.search(r'cleaned-dataset (\d+)', v).group(1)) 
                          for v in versions if re.search(r'cleaned-dataset (\d+)', v)]
        next_num = max(version_numbers, default=0) + 1
        version_name = f"cleaned-dataset {next_num}"
    
    # Create backup directory
    backup_path = os.path.join(BACKUP_DIR, version_name)
    os.makedirs(backup_path, exist_ok=True)
    
    # Copy files
    files_copied = 0
    for file in ["Train.jsonl", "Valid.jsonl", "Test.jsonl"]:
        src = os.path.join(DATA_DIR, file)
        dst = os.path.join(backup_path, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            files_copied += 1
            print(f"Copied {file} to {backup_path}")
    
    if files_copied == 0:
        print("No dataset files found to backup")
        return False
        
    # Create a metadata file with timestamp and info
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(backup_path, "metadata.txt"), "w") as f:
        f.write(f"Dataset backup created on: {timestamp}\n")
        f.write(f"Files included: {files_copied} files\n")
    
    print(f"Successfully backed up dataset as '{version_name}'")
    return True

def restore_dataset(version_name):
    """Restore a previously backed up dataset for training"""
    # Check if the specified version exists
    backup_path = os.path.join(BACKUP_DIR, version_name)
    if not os.path.exists(backup_path):
        print(f"Error: Dataset version '{version_name}' not found")
        return False
    
    # Create processed data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Copy files from backup to data/processed
    files_copied = 0
    for file in ["Train.jsonl", "Valid.jsonl", "Test.jsonl"]:
        src = os.path.join(backup_path, file)
        dst = os.path.join(DATA_DIR, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            files_copied += 1
            print(f"Restored {file} from {backup_path}")
    
    if files_copied == 0:
        print(f"No dataset files found in {version_name}")
        return False
    
    print(f"Successfully restored dataset version '{version_name}'")
    return True

def main():
    parser = argparse.ArgumentParser(description="Manage dataset versions for fine-tuning")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup the current dataset")
    backup_parser.add_argument("--name", type=str, help="Custom name for the backup (default: auto-numbered)")
    
    # List command
    subparsers.add_parser("list", help="List available dataset versions")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a dataset version")
    restore_parser.add_argument("version", type=str, help="Dataset version to restore")
    
    args = parser.parse_args()
    
    if args.command == "list":
        versions = list_dataset_versions()
        if versions:
            print("Available dataset versions:")
            for version in versions:
                # Check if metadata exists to show more info
                metadata_path = os.path.join(BACKUP_DIR, version, "metadata.txt")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = f.read().strip()
                    print(f"- {version}")
                    for line in metadata.split("\n"):
                        print(f"  {line}")
                else:
                    print(f"- {version}")
        else:
            print("No dataset versions found")
            
    elif args.command == "backup":
        backup_current_dataset(args.name)
        
    elif args.command == "restore":
        restore_dataset(args.version)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()