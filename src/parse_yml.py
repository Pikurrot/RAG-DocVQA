#!/usr/bin/env python
import sys
import yaml

"""
Usage:
  python parse_yml.py <yaml_file> <top_level_section> <key_in_section>
Example:
  python parse_yml.py multi_eval.yml data page_retrieval
This prints the array items (or single value) separated by spaces.
"""

if len(sys.argv) < 4:
    print("Usage: python parse_yml.py <yaml_file> <section> <key>")
    sys.exit(1)

yaml_file = sys.argv[1]
section = sys.argv[2]
key = sys.argv[3]

with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

values = config[section][key]
if isinstance(values, list):
    print(" ".join(str(v) for v in values))
else:
    print(values)
