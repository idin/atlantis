import os
import sys
import fnmatch

DEFAULT_IGNORE = [
    "__pycache__", "build", "dist", "*.pyc", "*.pyo", "*.egg-info", ".git", ".idea", ".vscode", "lessons"
]

def generate_tree(directory="atlantis", prefix="", ignore=None, is_root=True):
    """Recursively generates a tree structure of the given directory, avoiding duplication issues."""
    if ignore is None:
        ignore = DEFAULT_IGNORE

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    if is_root:
        print(f"📂 {os.path.basename(directory)}/")

    # Get sorted entries, excluding ignored ones
    entries = sorted(e for e in os.listdir(directory) if not any(fnmatch.fnmatch(e, pattern) for pattern in ignore))

    # Process each entry
    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "

        if os.path.isdir(path):
            print(f"{prefix}{connector}📂 {entry}/")
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_tree(path, new_prefix, ignore, is_root=False)
        else:
            print(f"{prefix}{connector}{entry}")

def main():
    """Runs the tree generator as a script or CLI command."""
    directory = "atlantis"  # Default to atlantis if no argument is provided
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    ignore_patterns = sys.argv[2:] if len(sys.argv) > 2 else None

    generate_tree(directory, ignore=ignore_patterns)

if __name__ == "__main__":
    main()
