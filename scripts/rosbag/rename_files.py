import os
import sys

def rename_files_in_directory(folder_path, new_name):
    # Use the last part of the folder path as the old prefix
    old_prefix = os.path.basename(os.path.normpath(folder_path))
    
    # Traverse all subdirectories and files in the given folder
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.startswith(old_prefix):
                # Replace old prefix with the new name
                new_filename = filename.replace(old_prefix, new_name, 1)
                
                old_filepath = os.path.join(dirpath, filename)
                new_filepath = os.path.join(dirpath, new_filename)
                
                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {old_filepath} -> {new_filepath}")

if __name__ == "__main__":
    # Check for proper usage
    if len(sys.argv) != 3:
        print("Usage: python rename_files.py <folder_path> <new_name>")
        sys.exit(1)

    folder_path = sys.argv[1]
    new_name = sys.argv[2]

    # Call the rename function
    rename_files_in_directory(folder_path, new_name)

