#!/usr/bin/env python3
"""
slim_analyzer.py

A standalone script for analyzing protein sequences for Short Linear Motifs (SLIMs)
using the ELM database motif patterns from eln2019.motifs.

Usage:
    python slim_analyzer.py --fasta protein.fasta --motifs eln2019.motifs --output slim_report.txt
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze protein sequences for Short Linear Motifs (SLIMs)"
    )
    parser.add_argument(
        "--fasta", 
        required=True,
        help="Path to the FASTA file containing protein sequence"
    )
    parser.add_argument(
        "--motifs", 
        required=True,
        help="Path to the motifs file (e.g., eln2019.motifs)"
    )
    parser.add_argument(
        "--output", 
        default="slim_report.txt",
        help="Path to the output report file (default: slim_report.txt)"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also output results in CSV format"
    )
    return parser.parse_args()


def read_fasta(fasta_file: str) -> Tuple[str, str]:
    """
    Read a FASTA file and return the sequence identifier and sequence.
    Handles simple FASTA files with one sequence.
    """
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    
    # Get the header line
    header = None
    sequence_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header is None:
                header = line
            else:
                # We only process the first sequence
                break
        else:
            if header is not None:
                sequence_lines.append(line)
    
    if header is None:
        raise ValueError(f"Invalid FASTA format: {fasta_file}")
    
    # Extract the sequence identifier from the header
    seq_id = header[1:].split()[0]
    
    # Combine all sequence lines
    sequence = ''.join(sequence_lines)
    
    return seq_id, sequence


def parse_motifs_file(motifs_file: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the motifs file and return a dictionary of motif ID to its properties.
    
    Format: MOTIF_ID REGEX_PATTERN # Description [instances]
    """
    motifs = {}
    
    with open(motifs_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Split the line into motif_id, pattern, and description
            parts = line.split('  ')
            if len(parts) < 2:
                print(f"Warning: Line {line_num} doesn't have enough fields: {line}", file=sys.stderr)
                continue
            
            motif_id = parts[0].strip()
            pattern = parts[1].strip()
            
            # Extract description if available
            description = ""
            if '#' in line:
                description = line.split('#', 1)[1].strip()
            
            # Determine motif type from the ID
            motif_type = "Unknown"
            if '_' in motif_id:
                motif_type = motif_id.split('_')[0]
            
            # Add the motif to the dictionary
            motifs[motif_id] = {
                'name': motif_id,
                'pattern': pattern,
                'description': description,
                'type': motif_type
            }
    
    return motifs


def find_motifs(sequence: str, motifs: Dict[str, Dict[str, str]]) -> List[Dict[str, any]]:
    """
    Find all motif instances in the given sequence.
    Returns a list of dictionaries with motif information and match positions.
    """
    results = []
    
    for motif_id, motif_info in motifs.items():
        pattern = motif_info['pattern']
        
        try:
            # Search for all matches of the pattern in the sequence
            for match in re.finditer(pattern, sequence):
                start_pos = match.start() + 1  # 1-indexed positions
                end_pos = match.end()
                match_seq = match.group(0)
                
                # Add the match information
                results.append({
                    'motif_id': motif_id,
                    'name': motif_info['name'],
                    'type': motif_info.get('type', ''),
                    'description': motif_info['description'],
                    'start': start_pos,
                    'end': end_pos,
                    'sequence': match_seq,
                    'pattern': pattern
                })
        except re.error:
            print(f"Warning: Invalid regex pattern for {motif_id}: {pattern}", file=sys.stderr)
    
    # Sort results by start position
    results.sort(key=lambda x: x['start'])
    
    return results


def write_report(results: List[Dict[str, any]], seq_id: str, sequence: str, output_file: str):
    """Write detailed report of found motifs to a text file."""
    with open(output_file, 'w') as f:
        f.write(f"SLIM Analysis Report for {seq_id}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Sequence length: {len(sequence)} amino acids\n")
        f.write(f"Total motifs found: {len(results)}\n\n")
        
        if not results:
            f.write("No motifs found in the sequence.\n")
            return
        
        # Group results by motif type
        motif_types = {}
        for result in results:
            motif_type = result['type']
            if motif_type not in motif_types:
                motif_types[motif_type] = []
            motif_types[motif_type].append(result)
        
        # Write summary by type
        f.write("Summary by motif type:\n")
        for motif_type, type_results in motif_types.items():
            f.write(f"  {motif_type}: {len(type_results)} instances\n")
        f.write("\n")
        
        # Write detailed results
        f.write("Detailed motif instances:\n")
        f.write("-" * 80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Motif {i}:\n")
            f.write(f"  ID: {result['motif_id']}\n")
            f.write(f"  Name: {result['name']}\n")
            f.write(f"  Type: {result['type']}\n")
            f.write(f"  Description: {result['description']}\n")
            f.write(f"  Position: {result['start']}-{result['end']}\n")
            f.write(f"  Sequence: {result['sequence']}\n")
            f.write(f"  Pattern: {result['pattern']}\n")
            
            # Show the motif in context (10 residues before and after)
            context_start = max(0, result['start'] - 11)
            context_end = min(len(sequence), result['end'] + 10)
            context = sequence[context_start:context_end]
            
            # Calculate the position of the match within the context
            match_start_in_context = result['start'] - context_start - 1
            match_end_in_context = result['end'] - context_start
            
            # Create a marker line to highlight the match
            marker = ' ' * match_start_in_context + '^' * (match_end_in_context - match_start_in_context) + ' ' * (len(context) - match_end_in_context)
            
            f.write(f"  Context: {context_start+1}-{context_end}\n")
            f.write(f"           {context}\n")
            f.write(f"           {marker}\n\n")


def write_csv_report(results: List[Dict[str, any]], seq_id: str, output_file: str):
    """Write results to a CSV file for easy import into other tools."""
    csv_file = output_file.replace('.txt', '.csv')
    
    with open(csv_file, 'w') as f:
        # Write header
        f.write("sequence_id,motif_id,motif_name,motif_type,start,end,match_sequence,description\n")
        
        # Write data rows
        for result in results:
            f.write(f"{seq_id},{result['motif_id']},{result['name']},{result['type']}," +
                    f"{result['start']},{result['end']},{result['sequence']}," +
                    f"\"{result['description']}\"\n")
    
    print(f"CSV report written to {csv_file}")


def main():
    args = parse_arguments()
    
    # Read the protein sequence from FASTA
    try:
        seq_id, sequence = read_fasta(args.fasta)
        print(f"Read sequence {seq_id}: {len(sequence)} amino acids")
    except Exception as e:
        print(f"Error reading FASTA file: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Parse the motifs file
    try:
        motifs = parse_motifs_file(args.motifs)
        print(f"Loaded {len(motifs)} motif patterns from {args.motifs}")
    except Exception as e:
        print(f"Error parsing motifs file: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Find motifs in the sequence
    results = find_motifs(sequence, motifs)
    print(f"Found {len(results)} motif instances in the sequence")
    
    # Write report
    write_report(results, seq_id, sequence, args.output)
    print(f"Report written to {args.output}")
    
    # Write CSV if requested
    if args.csv:
        write_csv_report(results, seq_id, args.output)


if __name__ == "__main__":
    main() 