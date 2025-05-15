#!/usr/bin/env python3
import os
import shutil

# Define the folder path to scan.
# You can update this variable to point to your desired folder.
FOLDER_PATH = "/workspaces/MUTE_v1/src"

def delete_identifier_items(folder_path):
    # Walk the directory tree from the bottom up to safely delete directories.
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Check for files ending with '.Identifier' and delete them.
        for file in files:
            if file.endswith(".Identifier"):
                file_path = os.path.join(root, file)
                print(f"Deleting file: {file_path}")
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

        # Check for directories ending with '.Identifier' and remove them.
        for dir_name in dirs:
            if dir_name.endswith(".Identifier"):
                dir_path = os.path.join(root, dir_name)
                print(f"Deleting directory: {dir_path}")
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {e}")

if __name__ == "__main__":
    if not os.path.exists(FOLDER_PATH):
        print(f"Error: Folder '{FOLDER_PATH}' does not exist.")
    else:
        delete_identifier_items(FOLDER_PATH)
        print("Deletion complete.")
