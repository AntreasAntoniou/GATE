import os
import shutil


def remove_pycache(root_folder):
    for foldername, _, filenames in os.walk(root_folder):
        if foldername.endswith("__pycache__"):
            print(f"Removing {foldername}")
            shutil.rmtree(foldername)


if __name__ == "__main__":
    root_folder = "."  # Current directory
    remove_pycache(root_folder)
