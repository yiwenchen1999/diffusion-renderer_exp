#!/usr/bin/env python3
"""
Create a sub-dataset from source_data_polyhaven using a priority list.

The priority list can be:
  1. A folder of PNG files (scene_name.png) - scene names from filenames
  2. A text file with one scene name per line

Usage:
  python create_priority_subset.py \
    --source ../LVSMExp/source_data_polyhaven \
    --priority_list priority_list \
    --output source_data_polyhaven_priority

  # Or with a text file:
  python create_priority_subset.py \
    --source ../LVSMExp/source_data_polyhaven \
    --priority_list priority_scenes.txt \
    --output source_data_polyhaven_priority
"""

import os
import argparse


def get_scene_names_from_folder(priority_dir):
    """Extract scene names from PNG filenames in priority_list folder."""
    if not os.path.isdir(priority_dir):
        return []
    names = set()
    for f in os.listdir(priority_dir):
        if f.endswith(".png"):
            names.add(os.path.splitext(f)[0])
    return sorted(names)


def get_scene_names_from_file(path):
    """Read scene names from text file, one per line."""
    if not os.path.isfile(path):
        return []
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def main():
    parser = argparse.ArgumentParser(description="Create priority sub-dataset")
    parser.add_argument("--source", required=True, help="Source data root (e.g. ../LVSMExp/source_data_polyhaven)")
    parser.add_argument("--priority_list", required=True,
                        help="Priority list: folder with .png files, or .txt with scene names")
    parser.add_argument("--output", default="source_data_polyhaven_priority",
                        help="Output sub-dataset directory")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of symlink (default: symlink)")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)

    if not os.path.isdir(source):
        print(f"Error: source {source} does not exist")
        return 1

    # Get priority scene names
    pl = args.priority_list
    if os.path.isdir(pl):
        scene_names = get_scene_names_from_folder(pl)
        print(f"Read {len(scene_names)} scenes from folder {pl}")
    elif os.path.isfile(pl):
        scene_names = get_scene_names_from_file(pl)
        print(f"Read {len(scene_names)} scenes from file {pl}")
    else:
        print(f"Error: priority_list {pl} not found")
        return 1

    if not scene_names:
        print("No scenes in priority list")
        return 1

    os.makedirs(output, exist_ok=True)
    import shutil

    created = 0
    skipped = 0
    for name in scene_names:
        src_dir = os.path.join(source, name)
        dst_dir = os.path.join(output, name)
        if not os.path.isdir(src_dir):
            print(f"  [SKIP] {name}: not found in source")
            skipped += 1
            continue
        if os.path.exists(dst_dir):
            continue
        try:
            if args.copy:
                shutil.copytree(src_dir, dst_dir)
            else:
                os.symlink(src_dir, dst_dir)
            created += 1
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
            skipped += 1

    print(f"\nCreated sub-dataset at {output}")
    print(f"  {created} scenes linked/copied, {skipped} skipped")
    print(f"\nRun experiment with:")
    print(f"  python run_polyhaven_experiment.py prepare --data_dir {output} --workspace polyhaven_workspace_priority")
    return 0


if __name__ == "__main__":
    exit(main())
