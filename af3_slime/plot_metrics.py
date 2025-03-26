import argparse
import json
import logging
import pathlib
import re
import shutil
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrosetta as pr
import seaborn as sns
import yaml
from config import PTMConfig
from contact_clustering import analyze_contact_patterns
from matplotlib.lines import Line2D
from plot_clusters import visualize_contact_clusters
from pycirclize import Circos
from pycirclize.utils import calc_group_spaces
from scipy.stats import sem
from score_interface import init_options, score_structures_parallel
from tqdm import tqdm
from utils import setup_logging

# Initialize PyRosetta with options before any parallel processing
pr.init(init_options, silent=True)

# Suppress PIL and matplotlib font warnings in log file
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


# Redirect scipy warnings to logging
def warning_to_log(message, category, filename, lineno, file=None, line=None):
    logging.getLogger("scipy").warning(f"{category.__name__}: {message}")


warnings.showwarning = warning_to_log


def add_ptm_line(ax, ptm_config: PTMConfig):
    """Add vertical line and legend for PTM"""
    if ptm_config is None:
        return None

    # Add vertical line and return it for legend
    line = ax.axvline(
        x=ptm_config.ptmPosition,
        color="purple",
        linestyle="--",
        alpha=0.5,
        label=f"{ptm_config.ptmType} at resi {ptm_config.ptmPosition}",
    )
    return line


def parse_cif_file(cif_path: pathlib.Path, atom_type: str) -> Dict[int, int]:
    """
    Parse CIF file to map atom numbers to residue numbers for specified atom type
    Returns: Dict[atom_number, residue_number] for chain A only
    """
    atom_to_res = {}
    with open(cif_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                parts = line.split()
                try:
                    atom_num = int(parts[1])  # atom id
                    atom_name = parts[3]  # atom type (CA, CB, etc)
                    chain = parts[6]  # chain ID
                    res_num = int(parts[8])  # residue number
                    if atom_name == atom_type and chain == "A":
                        atom_to_res[atom_num] = res_num
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse line: {line.strip()}")
                    continue

    # print(f"Found {len(atom_to_res)} chain A {atom_type} atoms in {cif_path}")
    return atom_to_res


def get_residue_range(folder_name: str) -> Tuple[int, int]:
    """
    Extract start and end residue numbers from folder name
    """
    # print(f"\nTrying to extract range from folder: {folder_name}")
    parts = folder_name.split("_")

    # Try different strategies:
    # 1. Look for consecutive numbers
    for i in range(len(parts) - 1):
        try:
            num1 = int(parts[i])
            num2 = int(parts[i + 1])
            # Sanity check: second number should be larger
            if num2 > num1:
                # print(f"Found range: {num1}-{num2}")
                return num1, num2
        except ValueError:
            continue

    # 2. Look for numbers with specific patterns
    pattern = r".*?(\d+)[_-](\d+).*"
    match = re.search(pattern, folder_name)
    if match:
        start, end = map(int, match.groups())
        if end > start:
            # print(f"Found range using regex: {start}-{end}")
            return start, end

    raise ValueError(f"Could not extract residue range from folder name: {folder_name}")


def collect_plddt_values(
    base_dir: pathlib.Path, atom_type: str
) -> Dict[int, List[float]]:
    """
    Collect pLDDT values for specified atom type across all prediction folders
    Returns: Dict[absolute_position, List[plddt_values]]
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Collecting pLDDT values for {atom_type} atoms")
    plddt_values = defaultdict(list)

    # Get all folders and sort them by start residue number
    folders = []
    for cif_path in base_dir.rglob("*_model.cif"):
        if "logs" in str(cif_path.parent):
            continue
        try:
            start_res, _ = get_residue_range(cif_path.parent.name)
            folders.append((start_res, cif_path))
        except ValueError as e:
            logger.debug(f"Skipping {cif_path.parent}: {e}")
            continue

    # Sort folders by start residue
    folders.sort()

    # Process folders in order
    for start_res, cif_path in folders:
        logger.debug(f"Processing: {cif_path}")
        folder = cif_path.parent
        try:
            start_res, end_res = get_residue_range(folder.name)

            # Parse CIF file
            atom_to_res = parse_cif_file(cif_path, atom_type)

            # Parse JSON file
            json_path = cif_path.with_name(
                cif_path.stem.replace("_model", "_confidences.json")
            )
            if not json_path.exists():
                logger.debug(f"Missing JSON file: {json_path}")
                continue

            with open(json_path) as f:
                data = json.load(f)

            # Map atom plddts to residue positions
            for atom_num, res_num in atom_to_res.items():
                abs_pos = start_res + res_num - 1
                plddt_values[abs_pos].append(data["atom_plddts"][atom_num - 1])

            logger.debug(f"Added pLDDT values for positions {start_res} to {end_res}")

        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Error processing {folder}: {e}")
            continue

    logger.info("Performing pLDDT calculations")
    logger.debug(f"Total positions collected: {len(plddt_values)}")
    logger.debug(
        f"Position ranges: {min(plddt_values.keys()) if plddt_values else 'None'} - {max(plddt_values.keys()) if plddt_values else 'None'}"
    )

    return plddt_values


def filter_ptm_matrices(
    token_res_ids: List[int], 
    contact_probs: np.ndarray, 
    pae: np.ndarray = None,
    average_tokens: bool = False,
    tokens_to_average: List[str] = None
) -> Tuple[List[int], np.ndarray, Optional[np.ndarray]]:
    """
    Filter matrices to handle duplicate tokens from PTM residues.
    
    Two modes:
    1. Keep only the second token (CA) - default behavior
    2. Average values across all tokens for PTM residues
    
    Args:
        token_res_ids: List of residue IDs for each token
        contact_probs: Contact probability matrix [N x N]
        pae: Optional PAE matrix [N x N]
        average_tokens: If True, average values for PTM residues instead of keeping just one token
        tokens_to_average: List of tokens to include in averaging (if None and average_tokens is True, all tokens will be averaged)
    
    Returns:
        Tuple of (filtered_token_res_ids, filtered_contact_probs, filtered_pae)
    """
    logger = logging.getLogger(__name__)
    
    if average_tokens:
        # If tokens_to_average is None or contains "ALL", average all tokens
        if tokens_to_average is None or "ALL" in tokens_to_average:
            logger.debug("Using token averaging mode for PTM residues (averaging ALL tokens)")
        else:
            logger.debug(f"Using token averaging mode for PTM residues (averaging tokens: {tokens_to_average})")
            
        # Group indices by residue ID
        residue_indices = {}
        for i, res_id in enumerate(token_res_ids):
            if res_id not in residue_indices:
                residue_indices[res_id] = []
            residue_indices[res_id].append(i)
        
        # For each residue with multiple tokens, average their values
        unique_res_ids = []
        indices_to_keep = []
        
        for res_id, indices in residue_indices.items():
            unique_res_ids.append(res_id)
            if len(indices) > 1:  # This is a PTM residue with multiple tokens
                # For now, we'll keep the first index as a placeholder
                # We'll replace its values with averaged values later
                indices_to_keep.append(indices[0])
                logger.debug(f"Residue {res_id} has {len(indices)} tokens - will average all")
            else:
                indices_to_keep.append(indices[0])
        
        # Create new matrices with unique residues
        n_unique = len(unique_res_ids)
        avg_contact_probs = np.zeros((n_unique, n_unique))
        avg_pae = None if pae is None else np.zeros((n_unique, n_unique))
        
        # Fill matrices with averaged values
        for i, res_i in enumerate(unique_res_ids):
            indices_i = residue_indices[res_i]
            for j, res_j in enumerate(unique_res_ids):
                indices_j = residue_indices[res_j]
                
                # Average contact probabilities
                values = []
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        values.append(contact_probs[idx_i, idx_j])
                avg_contact_probs[i, j] = np.mean(values)
                
                # Average PAE if provided
                if pae is not None:
                    pae_values = []
                    for idx_i in indices_i:
                        for idx_j in indices_j:
                            pae_values.append(pae[idx_i, idx_j])
                    avg_pae[i, j] = np.mean(pae_values)
        
        logger.debug(f"Original matrix shape: {contact_probs.shape}")
        logger.debug(f"Averaged matrix shape: {avg_contact_probs.shape}")
        logger.debug(f"Removed {len(token_res_ids) - len(unique_res_ids)} duplicate tokens through averaging")
        
        return unique_res_ids, avg_contact_probs, avg_pae
    
    else:
        # Original behavior: keep only the second token (CA) for PTM residues
        # Find duplicate residue positions (PTM residues)
        seen_positions = {}
        indices_to_keep = []
        
        for i, res_id in enumerate(token_res_ids):
            if res_id in seen_positions:
                # This is a duplicate (PTM residue)
                # We want to keep the second occurrence (CA)
                if seen_positions[res_id] == 1:  # This is the second occurrence
                    indices_to_keep.append(i)
                seen_positions[res_id] += 1
            else:
                # First time seeing this residue
                seen_positions[res_id] = 1
                indices_to_keep.append(i)
        
        # Create filtered token_res_ids
        filtered_token_res_ids = [token_res_ids[i] for i in indices_to_keep]
        
        # Filter contact probability matrix (both dimensions)
        filtered_contact_probs = contact_probs[indices_to_keep][:, indices_to_keep]
        
        # Filter PAE matrix if provided
        filtered_pae = None
        if pae is not None:
            filtered_pae = pae[indices_to_keep][:, indices_to_keep]
        
        logger.debug(f"Original matrix shape: {contact_probs.shape}")
        logger.debug(f"Filtered matrix shape: {filtered_contact_probs.shape}")
        logger.debug(f"Removed {len(token_res_ids) - len(filtered_token_res_ids)} duplicate tokens")
        
        return filtered_token_res_ids, filtered_contact_probs, filtered_pae


def collect_residue_contacts(
    base_dir: pathlib.Path,
    receptor_start: int = 1,
    average_tokens: bool = False,
    tokens_to_average: List[str] = None
) -> Dict[Tuple[int, int], List[float]]:
    logger = logging.getLogger(__name__)
    residue_contacts = defaultdict(list)

    # Add debug prints
    logger.debug(f"\nReceptor start offset: {receptor_start}")
    if average_tokens:
        if tokens_to_average is None:
            tokens_to_average = ["CA", "CB"]
        if "ALL" in tokens_to_average:
            logger.debug("Using token averaging for PTM residues (averaging ALL tokens)")
        else:
            logger.debug(f"Using token averaging for PTM residues (averaging tokens: {tokens_to_average})")
    else:
        logger.debug("Using second token (CA) for PTM residues")

    folders = []
    for json_path in base_dir.rglob("*_confidences.json"):
        try:
            start_res, end_res = get_residue_range(json_path.parent.name)
            folders.append((start_res, end_res, json_path))
        except ValueError as e:
            logger.debug(f"Skipping {json_path.parent}: {e}")
            continue

    folders.sort()

    for start_res, end_res, json_path in folders:
        try:
            with open(json_path) as f:
                data = json.load(f)

                if "contact_probs" in data and "token_res_ids" in data:
                    matrix = np.array(data["contact_probs"])
                    token_res_ids = data["token_res_ids"]
                    
                    # Filter out duplicate tokens
                    filtered_token_res_ids, filtered_matrix, _ = filter_ptm_matrices(
                        token_res_ids, 
                        matrix,
                        average_tokens=average_tokens,
                        tokens_to_average=tokens_to_average
                    )
                    
                    # Debug prints
                    logger.debug(f"\nProcessing file: {json_path}")
                    logger.debug(f"Original matrix shape: {matrix.shape}")
                    logger.debug(f"Filtered matrix shape: {filtered_matrix.shape}")
                    
                    peptide_length = end_res - start_res + 1
                    logger.debug(f"Peptide length: {peptide_length}")
                    logger.debug(f"Start res: {start_res}, End res: {end_res}")
                    
                    # For each binder residue
                    for i in range(peptide_length):
                        binder_pos = start_res + i
                        receptor_values = filtered_matrix[peptide_length:, i]
                        
                        # Debug first few receptor positions for first binder residue
                        if i == 0:
                            logger.debug(f"\nFirst few receptor positions for binder residue {binder_pos}:")
                            for j in range(min(5, len(receptor_values))):
                                receptor_pos = receptor_start + j
                                logger.debug(f"j={j}, receptor_pos={receptor_pos}, prob={receptor_values[j]:.3f}")

                        for j, prob in enumerate(receptor_values):
                            receptor_pos = receptor_start + j
                            residue_contacts[(binder_pos, receptor_pos)].append(prob)
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            continue

    # Debug final contact range
    if residue_contacts:
        receptor_positions = sorted(set(pos[1] for pos in residue_contacts.keys()))
        logger.debug(f"\nFinal receptor position range: {min(receptor_positions)}-{max(receptor_positions)}")
        logger.debug(f"Total unique receptor positions: {len(receptor_positions)}")

    return residue_contacts


def collect_contact_and_pae_data(
    base_dir: pathlib.Path,
    atom_type: str = "CA",
    do_relax: bool = False,
    average_tokens: bool = False,
    tokens_to_average: List[str] = None
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Collect contact probabilities and PAE data from all predictions.
    
    Args:
        base_dir: Base directory containing AF3 predictions
        atom_type: Atom type to use for contacts
        do_relax: Whether to use relaxed structures
        average_tokens: Whether to average values across all tokens for PTM residues
        tokens_to_average: List of tokens to include in averaging (if None or contains "ALL", all tokens will be averaged)
        
    Returns:
        Tuple of dictionaries mapping residue positions to lists of contact probabilities and PAE values
    """
    logger = logging.getLogger(__name__)
    
    # Log token handling strategy
    if average_tokens:
        if tokens_to_average is None:
            tokens_to_average = ["CA", "CB"]
        if "ALL" in tokens_to_average:
            logger.debug("Using token averaging for PTM residues (averaging ALL tokens)")
        else:
            logger.debug(f"Using token averaging for PTM residues (averaging tokens: {tokens_to_average})")
    else:
        logger.debug("Using second token (CA) for PTM residues")
        
    contact_probs = defaultdict(list)
    pae_data = defaultdict(list)

    # Get all folders and sort them by start residue number
    folders = []
    for json_path in base_dir.rglob("*_confidences.json"):
        if "summary" in str(json_path) or "logs" in str(json_path):
            continue

        try:
            start_res, end_res = get_residue_range(json_path.parent.name)
            folders.append((start_res, end_res, json_path))
        except ValueError as e:
            logger.debug(f"Skipping {json_path.parent}: {e}")
            continue

    folders.sort()

    for start_res, end_res, json_path in folders:
        logger.debug(f"Processing: {json_path}")
        try:
            peptide_length = end_res - start_res + 1

            with open(json_path) as f:
                data = json.load(f)

                # Process contact probabilities and PAE if token_res_ids exists
                if "token_res_ids" in data:
                    token_res_ids = data["token_res_ids"]
                    
                    # Process contact probabilities
                    if "contact_probs" in data:
                        matrix = np.array(data["contact_probs"])
                        # Get PAE matrix if available
                        pae_matrix = np.array(data["pae"]) if "pae" in data else None
                        
                        # Filter matrices
                        filtered_token_res_ids, filtered_matrix, filtered_pae = filter_ptm_matrices(
                            token_res_ids, 
                            matrix, 
                            pae_matrix,
                            average_tokens=average_tokens,
                            tokens_to_average=tokens_to_average
                        )
                        
                        # Process filtered contact probabilities
                        for i in range(peptide_length):
                            abs_pos = start_res + i
                            receptor_values = filtered_matrix[peptide_length:, i]

                            top_n = 3
                            top_contacts = np.sort(receptor_values)[-top_n:]
                            contact_probs[abs_pos].append(np.mean(top_contacts))

                        logger.debug(
                            f"Added contact probabilities for positions {start_res} to {end_res}"
                        )
                    else:
                        logger.debug(f"No contact_probs in {json_path}")

                    # Process filtered PAE
                    if filtered_pae is not None:
                        for i in range(peptide_length):
                            abs_pos = start_res + i
                            receptor_values = filtered_pae[peptide_length:, i]
                            pae_data[abs_pos].append(np.mean(receptor_values))
                        logger.debug(f"Added PAE for positions {start_res} to {end_res}")
                    else:
                        logger.debug(f"No pae in {json_path}")
                else:
                    logger.warning(f"No token_res_ids found in {json_path}")

        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error processing {json_path}: {e}")
            continue

    logger.info("Performing contact probabilities calculations")
    logger.debug("\nContact probabilities:")
    logger.debug(f"Total positions: {len(contact_probs)}")
    logger.debug(
        f"Position ranges: {min(contact_probs.keys()) if contact_probs else 'None'} - {max(contact_probs.keys()) if contact_probs else 'None'}"
    )

    logger.info("Performing PAE calculations")
    logger.debug("\nPAE data:")
    logger.debug(f"Total positions: {len(pae_data)}")
    logger.debug(
        f"Position ranges: {min(pae_data.keys()) if pae_data else 'None'} - {max(pae_data.keys()) if pae_data else 'None'}"
    )

    return contact_probs, pae_data


def plot_average_contacts(
    contact_probs: Dict[int, List[float]],
    output: str,
    ptm_configs: List[PTMConfig] = None,
):
    """Plot average contact probabilities with standard error of the mean"""
    positions = sorted(contact_probs.keys())
    means = []
    stds = []
    sems = []

    # Calculate means, stds and sems per position
    for pos in positions:
        values = contact_probs[pos]
        means.append(np.mean(values))
        stds.append(np.std(values))
        sems.append(sem(values))

    # Calculate threshold using position-wise averages
    threshold = np.mean(means) + np.mean(stds)

    """print("\nContact probabilities threshold calculation:")
    print(f"Average of position means: {np.mean(means):.2f}")
    print(f"Average of position SDs: {np.mean(stds):.2f}")
    print(f"Threshold (mean + 1SD): {threshold:.2f}")"""

    plt.figure(figsize=(12, 6))

    # Plot average line and sem area
    plt.plot(positions, means, label="Average Contact Probability", color="blue")
    plt.fill_between(
        positions,
        np.array(means) - np.array(sems),
        np.array(means) + np.array(sems),
        alpha=0.2,
        color="blue",
    )

    # Add threshold line
    plt.axhline(
        y=threshold,
        color="red",
        linestyle="-",
        label=f"Mean + 1SD ({threshold:.2f})",
        linewidth=0.5,
    )

    # Add PTM lines if configs provided
    if ptm_configs:
        for ptm_config in ptm_configs:
            add_ptm_line(plt.gca(), ptm_config)

    plt.xlabel("Residue Position")
    plt.ylabel("Contact Probability")
    plt.title("Average Contact Probabilities")
    plt.legend()
    # plt.grid(True)

    # Set y-axis limits to valid probability range
    plt.ylim(0, 1)

    plt.savefig(output)
    plt.close()


def plot_plddt_and_pae(
    plddt_values: Dict[int, List[float]],
    pae_data: Dict[int, List[float]],
    output: str,
    atom_type: str = "CA",
    ptm_configs: List[PTMConfig] = None,
):
    """Plot pLDDT and PAE on the same figure with dual y-axes"""

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot pLDDT on primary y-axis
    positions_plddt = sorted(plddt_values.keys())
    means_plddt = [np.mean(plddt_values[pos]) for pos in positions_plddt]
    sems_plddt = [sem(plddt_values[pos]) for pos in positions_plddt]

    ax1.set_xlabel("Residue Position")
    ax1.set_ylabel("pLDDT Score", color="blue")
    ax1.plot(
        positions_plddt, means_plddt, color="blue", label=f"Average pLDDT ({atom_type})"
    )
    ax1.fill_between(
        positions_plddt,
        np.array(means_plddt) - np.array(sems_plddt),
        np.array(means_plddt) + np.array(sems_plddt),
        alpha=0.2,
        color="blue",
    )
    ax1.tick_params(axis="y", labelcolor="blue")
    # ax1.grid(True)

    # Plot PAE on secondary y-axis
    ax2 = ax1.twinx()
    positions_pae = sorted(pae_data.keys())
    means_pae = [np.mean(pae_data[pos]) for pos in positions_pae]
    sems_pae = [sem(pae_data[pos]) for pos in positions_pae]

    ax2.set_ylabel("Predicted Aligned Error", color="red")
    ax2.plot(positions_pae, means_pae, color="red", label="Average PAE")
    ax2.fill_between(
        positions_pae,
        np.array(means_pae) - np.array(sems_pae),
        np.array(means_pae) + np.array(sems_pae),
        alpha=0.2,
        color="red",
    )
    ax2.tick_params(axis="y", labelcolor="red")

    # Add PTM lines if configs provided
    if ptm_configs:
        for ptm_config in ptm_configs:
            add_ptm_line(ax1, ptm_config)

    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Average pLDDT and PAE Values")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def collect_interface_metrics(
    base_dir: pathlib.Path, 
    pbar: tqdm = None,
    do_relax: bool = False
) -> Dict[int, Dict[str, List[float]]]:
    """
    Collect interface metrics across all prediction folders using parallel processing
    Args:
        base_dir: Directory containing prediction folders
        pbar: Progress bar (not used with parallel processing)
        do_relax: Whether to perform FastRelax before scoring
    Returns: Dict[absolute_position, Dict[metric_name, List[values]]]
    """
    logger = logging.getLogger(__name__)
    metrics_values = defaultdict(lambda: defaultdict(list))

    # Get all folders and sort them
    cif_files = []
    for cif_path in base_dir.rglob("*_model.cif"):
        if "logs" not in str(cif_path.parent):
            try:
                start_res, _ = get_residue_range(cif_path.parent.name)
                cif_files.append((start_res, str(cif_path)))
            except ValueError as e:
                logger.debug(f"Skipping {cif_path.parent}: {e}")
                continue

    # Sort by start residue
    cif_files.sort()
    structure_files = [f[1] for f in cif_files]
    start_residues = [f[0] for f in cif_files]

    # Score interfaces in parallel
    results = score_structures_parallel(
        structure_files=structure_files,
        binder_chain="A",
        do_relax=do_relax
    )

    # Process results and organize by position
    for (start_res, cif_path), result in zip(cif_files, results.values()):
        if result is None or result['scores'] is None:
            logger.error(f"Could not calculate interface scores for {cif_path}")
            continue

        try:
            start_res, end_res = get_residue_range(Path(cif_path).parent.name)
            scores = result['scores']

            # Add metric values to all positions in this peptide
            metrics_to_collect = [
                "interface_dG",
                "interface_sc",
                "interface_interface_hbonds",
            ]
            for pos in range(start_res, end_res + 1):
                for metric in metrics_to_collect:
                    metrics_values[pos][metric].append(scores[metric])

            logger.debug(f"Added interface metrics for positions {start_res} to {end_res}")

        except Exception as e:
            logger.error(f"Error processing {cif_path}: {e}")
            continue

    return metrics_values


def plot_interface_metrics(
    metrics_values: Dict[int, Dict[str, List[float]]],
    output_file: str,
    ptm_configs: List[PTMConfig] = None,
):
    """Plot interface metrics with multiple y-axes and SEM error bands"""
    
    positions = sorted(metrics_values.keys())

    # Calculate means and SEMs for each metric at each position
    mean_values = {
        metric: []
        for metric in ["interface_dG", "interface_sc", "interface_interface_hbonds"]
    }
    sem_values = {
        metric: []
        for metric in ["interface_dG", "interface_sc", "interface_interface_hbonds"]
    }

    for pos in positions:
        for metric in mean_values.keys():
            values = metrics_values[pos][metric]
            if values:
                mean_values[metric].append(np.mean(values))
                sem_values[metric].append(np.std(values) / np.sqrt(len(values)))
            else:
                mean_values[metric].append(np.nan)
                sem_values[metric].append(np.nan)

    # Create figure with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot interface_dG on left y-axis
    color1 = "black"
    ln1 = ax1.plot(
        positions, mean_values["interface_dG"], color=color1, label="Interface dG"
    )
    ax1.fill_between(
        positions,
        np.array(mean_values["interface_dG"]) - np.array(sem_values["interface_dG"]),
        np.array(mean_values["interface_dG"]) + np.array(sem_values["interface_dG"]),
        color=color1,
        alpha=0.3,
    )
    ax1.set_xlabel("Residue Position")
    ax1.set_ylabel("Interface dG (REU)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Create first right y-axis for shape complementarity
    ax2 = ax1.twinx()
    color2 = "#2ca02c"  # Green
    ln2 = ax2.plot(
        positions,
        mean_values["interface_sc"],
        color=color2,
        label="Shape Complementarity",
    )
    ax2.fill_between(
        positions,
        np.array(mean_values["interface_sc"]) - np.array(sem_values["interface_sc"]),
        np.array(mean_values["interface_sc"]) + np.array(sem_values["interface_sc"]),
        color=color2,
        alpha=0.3,
    )
    ax2.set_ylabel("Shape Complementarity (0-1)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Create second right y-axis for H-bonds
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    color3 = "#d62728"  # Red
    ln3 = ax3.plot(
        positions,
        mean_values["interface_interface_hbonds"],
        color=color3,
        label="Interface H-bonds",
    )
    ax3.fill_between(
        positions,
        np.array(mean_values["interface_interface_hbonds"])
        - np.array(sem_values["interface_interface_hbonds"]),
        np.array(mean_values["interface_interface_hbonds"])
        + np.array(sem_values["interface_interface_hbonds"]),
        color=color3,
        alpha=0.3,
    )
    ax3.set_ylabel("Number of H-bonds", color=color3)
    ax3.tick_params(axis="y", labelcolor=color3)

    # Add PTM lines if configs provided
    ptm_lines = []  # Store PTM lines for legend
    if ptm_configs:
        for ptm_config in ptm_configs:
            line = add_ptm_line(ax1, ptm_config)
            ptm_lines.append(line)

    # Add legend at the top including PTM lines
    lns = ln1 + ln2 + ln3 + ptm_lines  # Add PTM lines to legend
    labs = [leg.get_label() for leg in lns]
    ax1.legend(lns, labs, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()


def prepare_contact_dataframe(residue_contacts: Dict[Tuple[int, int], List[float]], 
                            probability_threshold: float = 0.3) -> pd.DataFrame:
    """Convert residue contacts dictionary to DataFrame format without averaging"""
    
    contact_data = []
    for (binder_pos, receptor_pos), probs in residue_contacts.items():
        # Create one row for each probability value
        for prob in probs:
            if prob >= probability_threshold:
                contact_data.append({
                    "binder_resi": binder_pos,
                    "receptor_resi": receptor_pos,
                    "contact_prob": prob
                })
    
    # Create DataFrame and sort by probability value
    df = pd.DataFrame(contact_data)
    df = df.sort_values("contact_prob", ascending=False)
    
    print("\nContact DataFrame Preview:")
    print(df.head(10))
    print(f"\nTotal contact entries: {len(df)}")
    
    return df


def plot_contact_circle(
    residue_contacts: Dict[Tuple[int, int], List[float]],
    output_path: str,
    ptm_configs: List[PTMConfig] = None,
    probability_threshold: float = 0.3,
    cluster_results = None,  # Add cluster_results parameter
):
    """Create circular diagram showing contacts using pycirclize"""
    logger = logging.getLogger(__name__)
    
    # Prepare contact data and normalize positions
    contact_data = []
    binder_positions = set()
    receptor_positions = set()
    
    # First collect all positions
    for (binder_pos, receptor_pos), probs in residue_contacts.items():
        binder_positions.add(binder_pos)
        receptor_positions.add(receptor_pos)
    
    # Log the binder positions we found
    logger.debug(f"Binder positions in residue_contacts: {sorted(binder_positions)}")
    logger.debug(f"Receptor positions in residue_contacts: {sorted(receptor_positions)}")
    
    # Create position mapping
    binder_map = {pos: idx for idx, pos in enumerate(sorted(binder_positions))}
    receptor_map = {pos: idx for idx, pos in enumerate(sorted(receptor_positions))}
    
    # Create normalized contact data
    for (binder_pos, receptor_pos), probs in residue_contacts.items():
        mean_prob = np.mean(probs)
        if mean_prob >= probability_threshold:
            contact_data.append({
                "binder_resi": binder_map[binder_pos],
                "receptor_resi": receptor_map[receptor_pos],
                "contact_prob": mean_prob,
                "contact_freq": len(probs)
            })
    
    df = pd.DataFrame(contact_data)
    df['contact_freq_norm'] = df['contact_freq'] / df['contact_freq'].max() if len(df) > 0 else 0
    
    # Calculate group spaces
    spaces = calc_group_spaces([1, 1], space_bw_group=15, space_in_group=2)
    
    # Initialize circos plot with normalized sector sizes
    sectors = {
        "Binder": len(binder_positions),
        "Receptor": len(receptor_positions)
    }
    circos = Circos(sectors, space=spaces)
    
    # Get the colormap for consistency across all visualizations
    contact_cmap = sns.color_palette("rocket_r", as_cmap=True)
    
    # Debug cluster results if available
    if cluster_results is not None:
        logger.debug("Cluster results available")
        if hasattr(cluster_results, 'positions'):
            logger.debug(f"Cluster positions: {cluster_results.positions[:10]}... (showing first 10)")
        if hasattr(cluster_results, 'dbscan_clusters'):
            logger.debug(f"DBSCAN clusters: {cluster_results.dbscan_clusters[:10]}... (showing first 10)")
            unique_clusters = np.unique(cluster_results.dbscan_clusters)
            logger.debug(f"Unique cluster IDs: {unique_clusters}")
        if hasattr(cluster_results, 'features'):
            logger.debug(f"Feature names: {cluster_results.feature_names}")
            # Find the index of LocalDensity in feature_names
            local_density_idx = cluster_results.feature_names.index('LocalDensity') if 'LocalDensity' in cluster_results.feature_names else 5
            logger.debug(f"LocalDensity index: {local_density_idx}")
    else:
        logger.debug("No cluster results provided")
    
    # Calculate receptor contact density
    # For each receptor position, calculate the average contact probability
    receptor_contact_density = np.zeros(len(receptor_positions))
    receptor_pos_list = sorted(receptor_positions)
    
    # Collect all contacts for each receptor position
    receptor_contacts = {pos: [] for pos in receptor_positions}
    for (binder_pos, receptor_pos), probs in residue_contacts.items():
        receptor_contacts[receptor_pos].extend(probs)
    
    # Calculate average contact probability for each receptor position
    for i, pos in enumerate(receptor_pos_list):
        if receptor_contacts[pos]:
            receptor_contact_density[i] = np.mean(receptor_contacts[pos])
    
    logger.debug(f"Receptor contact density stats: min={receptor_contact_density.min()}, max={receptor_contact_density.max()}, mean={receptor_contact_density.mean()}")
    logger.debug(f"Number of receptor positions with contact density > 0: {np.sum(receptor_contact_density > 0)}")
    
    # Fixed scale from 0 to 1 for all visualizations
    fixed_vmax = 1.0
    
    # Add tracks and labels with improved ticks
    for sector in circos.sectors:
        if sector.name == "Binder":
            # Create a single track for the gray area
            label_track = sector.add_track((70, 100))
            label_track.axis(fc="lightgray")
            
            # Get LocalDensity from cluster_results if available
            if cluster_results is not None and hasattr(cluster_results, 'features'):
                # Find the index of LocalDensity in feature_names
                local_density_idx = cluster_results.feature_names.index('LocalDensity') if 'LocalDensity' in cluster_results.feature_names else 5
                
                # Map cluster positions to binder positions
                cluster_pos_to_idx = {}
                for i, pos in enumerate(cluster_results.positions):
                    if pos in binder_map:
                        cluster_pos_to_idx[binder_map[pos]] = i
                
                # Create LocalDensity array for all binder positions
                local_density = np.zeros(len(binder_positions))
                
                # Fill in LocalDensity values where available
                for binder_idx, cluster_idx in cluster_pos_to_idx.items():
                    local_density[binder_idx] = cluster_results.features[cluster_idx, local_density_idx]
                
                logger.debug(f"LocalDensity stats: min={local_density.min()}, max={local_density.max()}, mean={local_density.mean()}")
                logger.debug(f"Number of positions with LocalDensity > 0: {np.sum(local_density > 0)}")
                
                # Create a heatmap for LocalDensity
                if np.any(local_density > 0):
                    logger.debug(f"Adding LocalDensity heatmap with {np.sum(local_density > 0)} positions")
                    
                    # Create a heatmap in the lower part of the track (70-85 radius range)
                    heatmap_track = sector.add_track((70, 85))
                    heatmap_track.axis(fc="lightgray")
                    
                    # Add heatmap to track
                    # We need to reshape the 1D array to a 2D array with 1 row
                    heatmap_data = local_density.reshape(1, -1)
                    heatmap_track.heatmap(
                        heatmap_data,
                        cmap=contact_cmap,  # Use the same colormap as contact probability
                        vmin=0,
                        vmax=fixed_vmax,  # Fixed scale from 0 to 1
                        rect_kws=dict(ec=None, lw=0)  # Remove edges to eliminate white spaces
                    )
                    
                    # Add a title for the LocalDensity
                    group_deg_lim = circos.get_group_sectors_deg_lim([sector.name])
                    group_center_deg = sum(group_deg_lim) / 2
                    #circos.text("LocalDensity", r=77.5, deg=group_center_deg, size=8, adjust_rotation=True)
                else:
                    logger.debug("No LocalDensity data to display in heatmap")
            else:
                logger.debug("No cluster results available for LocalDensity visualization")
            
            # Add ticks every 5 positions with labels (in the upper part of the track)
            label_track = sector.add_track((85, 100))
            label_track.axis(fc="lightgray")
            label_track.xticks_by_interval(
                interval=5,
                label_formatter=lambda v: str(sorted(binder_positions)[int(v)]),  # Convert index back to actual residue number
                label_orientation="vertical",
                label_size=8,
                tick_length=3,
                line_kws=dict(linewidth=1.5)
            )
            
            # Add PTM markers to binder sector
            if ptm_configs:
                for ptm in ptm_configs:
                    if ptm.ptmPosition in binder_map:
                        norm_pos = binder_map[ptm.ptmPosition]
                        # Draw vertical line spanning both tracks
                        label_track.line(
                            x=[norm_pos, norm_pos],  # Same X position for start and end
                            y=[0, 1],  # Vertical range (from inner to outer radius)
                            color="purple",
                            ls="-",
                            lw=1.7,
                        )
                        # Also add line to heatmap track if it exists
                        if 'heatmap_track' in locals():
                            heatmap_track.line(
                                x=[norm_pos, norm_pos],  # Same X position for start and end
                                y=[0, 1],  # Vertical range (from inner to outer radius)
                                color="purple",
                                ls="-",
                                lw=1.7,
                            )
        else:  # Receptor sector
            # Create a single track for the gray area
            label_track = sector.add_track((70, 100))
            label_track.axis(fc="lightgray")
            
            # Create a heatmap for receptor contact density
            if np.any(receptor_contact_density > 0):
                logger.debug(f"Adding receptor contact density heatmap with {np.sum(receptor_contact_density > 0)} positions")
                
                # Create a heatmap in the lower part of the track (70-85 radius range)
                heatmap_track = sector.add_track((70, 85))
                heatmap_track.axis(fc="lightgray")
                
                # Add heatmap to track
                # We need to reshape the 1D array to a 2D array with 1 row
                heatmap_data = receptor_contact_density.reshape(1, -1)
                heatmap_track.heatmap(
                    heatmap_data,
                    cmap=contact_cmap,  # Use the same colormap as contact probability
                    vmin=0,
                    vmax=fixed_vmax,  # Fixed scale from 0 to 1
                    rect_kws=dict(ec=None, lw=0)  # Remove edges to eliminate white spaces
                )
                
                # Add a title for the receptor contact density
                group_deg_lim = circos.get_group_sectors_deg_lim([sector.name])
                group_center_deg = sum(group_deg_lim) / 2
                #circos.text("Contact Density", r=77.5, deg=group_center_deg, size=8, adjust_rotation=True)
            else:
                logger.debug("No receptor contact density data to display in heatmap")
            
            # Add ticks every 5 positions with labels (in the upper part of the track)
            label_track = sector.add_track((85, 100))
            label_track.axis(fc="lightgray")
            
            positions = sorted(receptor_positions)
            
            # Calculate appropriate interval based on number of positions
            # Use a smaller interval for fewer positions, larger for many positions
            num_positions = len(positions)
            if num_positions <= 10:
                interval = 1  # Show every position for small sets
            elif num_positions <= 30:
                interval = 2  # Show every other position for medium sets
            else:
                interval = 5  # Show every fifth position for large sets
                
            logger.debug(f"Receptor has {num_positions} positions, using interval {interval}")
            
            # Add ticks with safe index handling
            label_track.xticks_by_interval(
                interval=interval,
                label_formatter=lambda v: str(positions[min(int(v), len(positions)-1)]),  # Safely convert index to residue number
                label_orientation="vertical",
                label_size=8,
                tick_length=3,
                line_kws=dict(linewidth=1.5)
            )

    # Add group labels
    for idx, group in enumerate(["Binder", "Receptor"]):
        group_deg_lim = circos.get_group_sectors_deg_lim([group])
        group_center_deg = sum(group_deg_lim) / 2
        circos.text(group, r=115, deg=group_center_deg, size=10, adjust_rotation=True)
    
    # Create links
    for _, row in df.iterrows():
        region1 = ("Binder", row['binder_resi'], row['binder_resi'] + 1)
        region2 = ("Receptor", row['receptor_resi'], row['receptor_resi'] + 1)
        
        # Fix deprecated get_cmap
        color = contact_cmap(row['contact_prob'])
        
        circos.link(
            region1, 
            region2,
            color=color,
            linewidth=row['contact_freq_norm']*2,
            alpha=row['contact_prob']**3
        )
    
    # Add colorbar for contact probability
    circos.colorbar(
        bounds=(1.1, 0.3, 0.02, 0.4),  # Right side position (x, y, width, height)
        vmin=0,  # Start from 0 for full scale
        vmax=fixed_vmax,  # Fixed scale to 1.0
        cmap=contact_cmap,
        label="Contact Probability"
    )
    
    # First call plotfig()
    fig = circos.plotfig()
    
    # We don't need separate colorbars since we're using the same colormap and scale
    # Just add the PTM legend if configs exist
    if ptm_configs:
        legend_handles = [
            Line2D([], [], color="purple", ls="-", lw=1.7,
                  label=f"{ptm.ptmType} at {ptm.ptmPosition}")
            for ptm in ptm_configs
        ]
        ptm_legend = circos.ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.15, 0.8),
            loc="center",
            fontsize=8
        )
    
    logger.debug(f"Saving circos plot to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_banner():
    banner = """
    

    █████╗ ███████╗ 🔍🔍🔍🔍       ███████╗██╗     ██╗███╗   ███╗███████╗ 
   ██╔══██╗██╔════╝  🔍🔍🔍🔍      ██╔════╝██║     ██║████╗ ████║██╔════╝   
   ███████║█████╗     🔍🔍🔍🔍     ███████╗██║     ██║██╔████╔██║█████╗       
   ██╔══██║██╔══╝      🔍🔍🔍🔍    ╚════██║██║     ██║██║╚██╔╝██║██╔══╝         
   ██║  ██║██║          🔍🔍🔍🔍   ███████║███████╗██║██║ ╚═╝ ██║███████╗        
   ╚═╝  ╚═╝╚═╝           🔍🔍🔍🔍  ╚══════╝╚══════╝╚═╝╚═╝     ╚═╝╚══════╝ ANALYSIS 🔍      
                              
          [ AlphaFold3 SLIM analysis pipeline ]
          
    
    """
    print(banner)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    If no file is provided, return empty configuration.
    """
    logger = logging.getLogger(__name__)
    
    # Start with empty configuration
    config = {}
    
    # If user config is provided, load it
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                yaml_content = f.read()
                config = yaml.safe_load(yaml_content)
                logger.debug(f"Loaded configuration from {config_path}")
                
                # Log the YAML file contents
                logger.debug("YAML Configuration:")
                for line in yaml_content.splitlines():
                    logger.debug(f"  {line}")

                logger.info(f"YAML Configuration file: {config_path}")

        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
    
    return config


def apply_config_to_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply configuration values to args, but only if they weren't explicitly set via command line.
    This preserves command line argument priority.
    """
    # Create a copy of args to avoid modifying the original
    updated_args = argparse.Namespace(**vars(args))
    logger = logging.getLogger(__name__)
    
    # Map config values to args if they exist in config
    if 'input' in config:
        if 'dir' in config['input'] and args.dir == ".":  # Only update if using default
            updated_args.dir = config['input']['dir']
            logger.debug(f"Setting base directory from config: {updated_args.dir}")
        if 'ptm_config' in config['input'] and args.ptm_config is None:
            updated_args.ptm_config = Path(config['input']['ptm_config']) if config['input']['ptm_config'] else None
            logger.debug(f"Setting PTM config from config: {updated_args.ptm_config}")
    
    if 'output' in config:
        if 'metrics' in config['output'] and args.metrics_output == "average_metrics.png":
            updated_args.metrics_output = config['output']['metrics']
            logger.debug(f"Setting metrics output from config: {updated_args.metrics_output}")
        if 'contacts' in config['output'] and args.contacts_output == "average_contacts.png":
            updated_args.contacts_output = config['output']['contacts']
            logger.debug(f"Setting contacts output from config: {updated_args.contacts_output}")
        if 'interface_metrics' in config['output'] and args.interface_metrics_output is None:
            updated_args.interface_metrics_output = config['output']['interface_metrics']
            logger.debug(f"Setting interface metrics output from config: {updated_args.interface_metrics_output}")
        if 'chord_diagram' in config['output'] and args.chord_diagram is None:
            updated_args.chord_diagram = config['output']['chord_diagram']
            logger.debug(f"Setting chord diagram output from config: {updated_args.chord_diagram}")
    
    if 'processing' in config:
        if 'atom_type' in config['processing'] and args.atom_type == "CA":
            updated_args.atom_type = config['processing']['atom_type']
            logger.debug(f"Setting atom type from config: {updated_args.atom_type}")
        if 'do_relax' in config['processing'] and not args.do_relax:  # Only update if not set via command line
            updated_args.do_relax = config['processing']['do_relax']
            logger.debug(f"Setting do_relax from config: {updated_args.do_relax}")
        if 'receptor' in config['processing'] and 'start' in config['processing']['receptor'] and args.receptor_start == 1:
            updated_args.receptor_start = config['processing']['receptor']['start']
            logger.debug(f"Setting receptor start from config: {updated_args.receptor_start}")
    
    # Handle PTM token filtering options
    if 'parameters' in config and 'ptm_filtering' in config['parameters']:
        ptm_filtering = config['parameters']['ptm_filtering']
        if 'average_tokens' in ptm_filtering and not args.average_tokens:  # Only update if not set via command line
            updated_args.average_tokens = ptm_filtering['average_tokens']
            logger.debug(f"Setting average_tokens from config: {updated_args.average_tokens}")
    
    return updated_args


def main():
    # Initialize a basic logger at the beginning of the function
    logger = logging.getLogger(__name__)

    # Record start time
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="Plot average metrics")
    parser.add_argument(
        "--dir", type=str, default=".", help="Directory containing prediction folders"
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="average_metrics.png",
        help="Output plot filename for pLDDT and PAE",
    )
    parser.add_argument(
        "--contacts-output",
        type=str,
        default="average_contacts.png",
        help="Output plot filename for contacts",
    )
    parser.add_argument(
        "--interface-metrics-output",
        type=str,
        help="Output plot filename for interface metrics (optional)",
        default=None
    )
    parser.add_argument(
        "--cluster-output",
        type=str,
        default="clusters",
        help="Prefix for cluster analysis output files",
    )
    parser.add_argument(
        "--atom-type",
        type=str,
        default="CA",
        help="Atom type to analyze for pLDDT (e.g., CA, CB)",
    )
    parser.add_argument(
        "--do-relax",
        action="store_true",
        help="Perform FastRelax before interface scoring",
    )
    parser.add_argument(
        "--ptm-config", type=Path, help="Path to PTM configuration YAML file"
    )
    parser.add_argument(
        "--receptor-start",
        type=int,
        default=1,
        help="Starting residue number for receptor sequence",
    )
    parser.add_argument(
        "--chord-diagram", type=str, help="Output HTML file for contact chord diagram"
    )
    parser.add_argument(
        "--average-tokens",
        action="store_true",
        help="Average values across all tokens for PTM residues instead of keeping just one",
    )
    parser.add_argument(
        "--config", type=Path, help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    # Initialize tokens_to_average with ALL to average all tokens
    tokens_to_average = ["ALL"]
    
    # Get timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # First, fully resolve the base directory from config if provided
    base_dir = pathlib.Path(args.dir)
    
    # Load configuration if provided to get the correct base directory
    if args.config:
        logger.debug(f"Loading configuration from {args.config}")
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                if 'input' in config and 'dir' in config['input'] and args.dir == ".":
                    args.dir = config['input']['dir']
                    logger.debug(f"Setting base directory from config: {args.dir}")
                    base_dir = pathlib.Path(args.dir)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Now create plots directory in the fully resolved base directory
    plots_dir = base_dir / "plots"
    try:
        plots_dir.mkdir(exist_ok=True)
        logger.debug(f"Created plots directory at: {plots_dir.resolve()}")
    except Exception as e:
        logger.error(f"Error creating plots directory: {e}")
        logger.error("Will use current directory for outputs")
        plots_dir = pathlib.Path(".")

    # Setup logging with timestamp - AFTER resolving the base directory
    logger = setup_logging(plots_dir, timestamp)
    
    # Now we can use the logger
    logger.debug(f"Created plots directory at: {plots_dir.resolve()}")
    
    # Now load the full configuration and apply all settings
    if args.config:
        logger.info(f"Loading YAML configuration: {args.config}")
        config = load_config(args.config)
        # Update args with config values
        args = apply_config_to_args(config, args)
        logger.debug(f"Configuration applied: {vars(args)}")
        
        # Get tokens_to_average from config if available, otherwise keep ["ALL"]
        if 'parameters' in config and 'ptm_filtering' in config['parameters']:
            ptm_filtering = config['parameters']['ptm_filtering']
            if 'tokens_to_average' in ptm_filtering:
                # If config specifies tokens but we want all, keep ["ALL"]
                if ptm_filtering['tokens_to_average'] != ["ALL"]:
                    logger.debug(f"Config specified tokens {ptm_filtering['tokens_to_average']}, but overriding to use ALL tokens")
                tokens_to_average = ["ALL"]
                logger.debug(f"Using tokens_to_average: {tokens_to_average}")

        # Copy the config file to the plots directory
        shutil.copy(args.config, plots_dir / args.config.name)
    
    # Update base_dir with potentially updated args.dir (should be the same now)
    base_dir = pathlib.Path(args.dir)
    logger.info(f"Base directory: {base_dir}")
    
    # No need to update plots directory again since we already created it in the correct location
    logger.info(f"Plots directory: {plots_dir}")

    # Log token handling strategy
    if args.average_tokens:
        logger.info("Using token averaging for PTM residues (averaging ALL tokens)")
    else:
        logger.info("Using second token (CA) for PTM residues")

    logger.info("Starting analysis...")

    # Load PTM configs if provided
    ptm_configs = []
    if args.ptm_config:
        ptm_config_path = args.ptm_config
        # If ptm_config is not an absolute path, look for it where we run the script
        if not ptm_config_path.is_absolute():
            ptm_config_path = ptm_config_path
        
        if ptm_config_path.exists():
            with open(ptm_config_path) as f:
                ptm_data = yaml.safe_load(f)
                if "modifications" in ptm_data:
                    for ptm_info in ptm_data["modifications"]:
                        ptm_config = PTMConfig(
                            ptmType=ptm_info["ptmType"], ptmPosition=ptm_info["ptmPosition"]
                        )
                        ptm_configs.append(ptm_config)
                        logger.info(
                            f"Loaded PTM configuration: {ptm_config.ptmType} at position {ptm_config.ptmPosition}"
                        )
        else:
            logger.warning(f"PTM config file not found: {ptm_config_path}")

    # Update output paths to use plots directory
    metrics_output = plots_dir / args.metrics_output
    contacts_output = plots_dir / args.contacts_output
    if args.interface_metrics_output:
        interface_metrics_output = plots_dir / args.interface_metrics_output
    cluster_output = plots_dir / args.cluster_output

    # Collect data with progress messages
    logger.info("Collecting pLDDT values...")
    plddt_values = collect_plddt_values(base_dir, args.atom_type)

    logger.info("Collecting contact and PAE data...")
    contact_probs, pae_data = collect_contact_and_pae_data(
        base_dir, 
        atom_type=args.atom_type,
        do_relax=args.do_relax,
        average_tokens=args.average_tokens,
        tokens_to_average=tokens_to_average
    )

    # Only check relaxation status if we're calculating interface metrics
    if args.interface_metrics_output:
        # Check relaxation status of first structure to determine general approach
        first_cif = next(base_dir.rglob("*_model.cif"))
        first_relaxed = str(first_cif).replace(".cif", "_converted_relaxed.pdb")

        if args.do_relax:
            if Path(first_relaxed).exists():
                logger.info("Found pre-relaxed structures")
                logger.info("Using pre-relaxed structures for interface analysis")
            else:
                logger.info("Performing relaxation before interface analysis")
        else:
            logger.info("Using original structures without relaxation")

    
    plot_plddt_and_pae(
        plddt_values, pae_data, metrics_output, args.atom_type, ptm_configs
    )
    plot_average_contacts(contact_probs, contacts_output, ptm_configs)
    
    # Only calculate and plot interface metrics if output path is specified
    if args.interface_metrics_output:
        logger.info("Collecting interface metrics...")
        interface_metrics = collect_interface_metrics(base_dir, None, args.do_relax)

        logger.info("Generating plots...")
        
        # Plot interface metrics only if we calculated them
        plot_interface_metrics(interface_metrics, interface_metrics_output, ptm_configs)

    logger.info("Performing cluster analysis...")
    cluster_results = analyze_contact_patterns(contact_probs)
    visualize_contact_clusters(cluster_results, cluster_output)

    # Generate chord diagram if requested
    if args.chord_diagram:
        logger.info("Generating contact density diagram...")
        residue_contacts = collect_residue_contacts(
            base_dir=base_dir, 
            receptor_start=args.receptor_start,
            average_tokens=args.average_tokens,
            tokens_to_average=tokens_to_average
        )
        
        # Force clear any existing figure
        plt.close('all')

        # Update output path to use circle_plots directory
        circle_plot_output = plots_dir / Path(args.chord_diagram).name
        
        # Call our density plot function with updated path
        plot_contact_circle(
            residue_contacts=residue_contacts,
            output_path=circle_plot_output,
            ptm_configs=ptm_configs,
            probability_threshold=0.3,
            cluster_results=cluster_results,  # Pass the cluster results
        )

    # Calculate total execution time
    execution_time = datetime.now() - start_time
    hours, remainder = divmod(execution_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    logger.info("Analysis complete!")
    logger.info(f"Total execution time: {time_str}")
    logger.info(f"Please check the log file in {plots_dir}/{timestamp}.log")


if __name__ == "__main__":
    print_banner()
    main()

