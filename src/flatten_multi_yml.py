#!/usr/bin/env python
import sys
import yaml

"""
Usage:
python flatten_multi_yml.py multi_eval.yml

This script prints:

1) A line containing the GPU array (if any) from "runtime.visible_devices".
If no such key is found or it's not a list, prints "NO_VISIBLE_DEVICES".

2) A line containing all the "multi-run" keys, space-separated.
If there are no multi-run keys, prints "NO_MULTI_PARAMS".

3) A line with the integer N = length of those lists (or 0 if no multi-run keys).

4) For i in [0..N-1], a line with the i-th values from each multi-run key, space-separated.

Definition of "multi-run" parameter:
- It's a flattened key whose value is a list (e.g. [Oracle, Concat, ...]).
- We exclude the "runtime.visible_devices" key from multi-run, because we treat that specially.
- All multi-run lists must have the same length.

Any scalar or boolean remains a single value in your YAML; it won't appear here as a multi-run parameter.
"""

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            short_key = new_key.split(sep)[-1]
            items[short_key] = v
    return items

def main():
    if len(sys.argv) < 2:
        print("Usage: python flatten_multi_yml.py <yaml_file>")
        sys.exit(1)

    yaml_file = sys.argv[1]
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    flat = flatten_dict(config)

    # GPU key handling
    gpu_key = "visible_devices"
    # Because we removed the prefix, if it was "runtime.visible_devices", now it's just "visible_devices"
    gpu_list = None
    if gpu_key in flat and isinstance(flat[gpu_key], list):
        gpu_list = flat.pop(gpu_key)
    else:
        gpu_list = None

    # Identify multi-run keys (where value is a list)
    multi_keys = []
    multi_lists = []
    for k, v in flat.items():
        if isinstance(v, list):
            multi_keys.append(k)
            multi_lists.append(v)

    # Print GPU array or placeholder
    if gpu_list is not None:
        print(" ".join(str(g) for g in gpu_list))
    else:
        print("NO_VISIBLE_DEVICES")

    # If no multi-run keys
    if not multi_keys:
        print("NO_MULTI_PARAMS")
        print("0")
        sys.exit(0)

    # Ensure all multi-lists have same length
    lengths = [len(lst) for lst in multi_lists]
    if len(set(lengths)) != 1:
        print(f"ERROR: Not all multi-lists have the same length:\n{multi_keys} => {lengths}")
        sys.exit(1)

    N = lengths[0]

    # Print multi-run keys
    print(" ".join(multi_keys))
    # Print length
    print(N)
    # Print each row
    for i in range(N):
        row_values = []
        for lst in multi_lists:
            row_values.append(str(lst[i]))
        print(" ".join(row_values))

if __name__ == "__main__":
    main()
