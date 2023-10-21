from pathlib import Path

from fire import Fire


def write_code_to_file(directory=".", output_file="output.txt"):
    with open(output_file, "w") as f_out:
        # Path provides the rglob method that is handy in getting
        # all the .py files in the directory and its subdirectories
        for file in Path(directory).rglob("*.py"):
            f_out.write(f"\nFile: {file}\n\n")
            with open(file, "r") as f_in:
                lines = f_in.readlines()
                f_out.writelines(lines)


if __name__ == "__main__":
    Fire(write_code_to_file)
