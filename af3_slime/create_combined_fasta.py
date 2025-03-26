#!/usr/bin/env python3

"""
create_combined_fasta.py

This script generates a combined FASTA file containing all peptide fragments
in order according to their positions from AlphaFold 3 input JSONs.
"""

import argparse
import json
import os
import re
import sys
from typing import List, Tuple


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a combined FASTA file from AlphaFold 3 input JSONs"
    )
    parser.add_argument(
        "--json-dir", 
        required=True,
        help="Path to the directory containing AlphaFold 3 input JSON files"
    )
    parser.add_argument(
        "--output", 
        default="combined_fragments.fasta",
        help="Path to the output FASTA file (default: combined_fragments.fasta)"
    )
    parser.add_argument(
        "--prefix",
        default="fragment",
        help="Prefix to use for fragment headers (default: fragment)"
    )
    return parser.parse_args()


def extract_position_from_filename(filename: str) -> Tuple[int, int]:
    """
    Extract start and end positions from a JSON filename.
    Assumes format with positions in the filename.
    """
    # Match patterns like "388_397" in filenames like "shp2_ncSH2_ctir_388_397.json"
    match = re.search(r'(\d+)_(\d+)', filename)
    if match:
        start_pos = int(match.group(1))
        end_pos = int(match.group(2))
        return start_pos, end_pos
    else:
        # If no position info in filename, return None
        return None


def get_fragments_from_jsons(json_dir: str) -> List[Tuple[int, int, str]]:
    """
    Extract peptide fragments from JSON files in the specified directory.
    Returns a list of tuples: (start_position, end_position, sequence)
    """
    fragments = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        try:
            # Full path to the JSON file
            json_path = os.path.join(json_dir, json_file)
            
            # Extract positions from filename
            positions = extract_position_from_filename(json_file)
            
            if positions is None:
                # Skip files that don't have position information
                continue
                
            start_pos, end_pos = positions
            
            # Read and parse the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract the sequence for protein with id "A" (the peptide fragment)
            sequence = None
            if 'sequences' in data:
                for seq_entry in data['sequences']:
                    if 'protein' in seq_entry and seq_entry['protein'].get('id') == 'A':
                        sequence = seq_entry['protein'].get('sequence', '')
                        if sequence:
                            fragments.append((start_pos, end_pos, sequence))
                        break
            
            if sequence is None:
                print(f"Warning: No sequence found for chain A in {json_file}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}", file=sys.stderr)
    
    # Sort fragments by start position
    fragments.sort(key=lambda x: x[0])
    
    return fragments


def write_fasta(output_file: str, fragments: List[Tuple[int, int, str]], prefix: str):
    """
    Write the combined FASTA file with fragments.
    """
    with open(output_file, 'w') as f:
        # Write each fragment
        for start_pos, end_pos, sequence in fragments:
            f.write(f">{prefix}_{start_pos}_{end_pos}\n")
            # Write sequence with 80 characters per line
            for i in range(0, len(sequence), 80):
                f.write(f"{sequence[i:i+80]}\n")


def main():
    args = parse_arguments()
    
    # Get fragments from JSON files
    fragments = get_fragments_from_jsons(args.json_dir)
    
    if not fragments:
        print("No fragments found in the specified directory.", file=sys.stderr)
        sys.exit(1)
    
    # Write the combined FASTA file
    write_fasta(args.output, fragments, args.prefix)
    
    print(f"Combined FASTA file created: {args.output}")
    print(f"Contains {len(fragments)} fragments.")


if __name__ == "__main__":
    main()