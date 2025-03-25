# adapted from https://github.com/martinpacesa/BindCraft/blob/main/functions/pyrosetta_utils.py

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pyrosetta as pr
from Bio.PDB import PDBIO, MMCIFParser, PDBParser, Selection
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Three letter to one letter amino acid code mapping
three_to_one_map = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

# At the top of the file, after imports
init_options = "-ignore_unrecognized_res \
    -load_PDB_components false \
    -ignore_waters false \
    -holes:dalphaball /home/tiagogomes/software/bindcraft/functions/DAlphaBall.gcc \
    -multiple_processes_writing_to_one_directory true \
    -run:multiple_processes_writing_to_one_directory \
    -jd3:nthreads 20 \
    -packstat:threads 20 \
    -mute all "


# Initialize PyRosetta once
pr.init(init_options, silent=True)


def convert_cif_to_pdb(cif_file):
    """Convert CIF file to PDB format using Biopython"""
    # Parse CIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)

    # Create output PDB filename
    pdb_file = os.path.splitext(cif_file)[0] + "_converted.pdb"

    # Save as PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)

    logger.debug(f"Converted CIF to PDB: {pdb_file}")
    return pdb_file


def hotspot_residues(trajectory_file, binder_chain="A", atom_distance_cutoff=4.0):
    """Identify interface residues from structure file"""
    # Choose parser based on file extension
    if trajectory_file.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("complex", trajectory_file)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], "A")
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]["A"], "A")
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues


def score_interface(
    structure_file: str, binder_chain: str = "A", do_relax: bool = False
):
    """
    Score interface from structure file
    Args:
        structure_file: Path to structure file (CIF or PDB)
        binder_chain: Chain ID of the binder (default: "A")
        do_relax: Whether to perform FastRelax before scoring (default: False)
    """
    logger.debug(f"Starting interface scoring for {structure_file}")

    # Convert CIF to PDB if needed
    if structure_file.endswith(".cif"):
        pdb_file = convert_cif_to_pdb(structure_file)
    else:
        pdb_file = structure_file

    # Create paths for relaxed structure
    base_path = os.path.splitext(pdb_file)[0]
    relaxed_pdb_path = f"{base_path}_relaxed.pdb"

    try:
        logger.debug(f"Processing structure: {pdb_file}")

        pose = pr.pose_from_pdb(pdb_file)

        # Perform relaxation if requested
        if do_relax:
            if os.path.exists(relaxed_pdb_path):
                logger.debug(f"Using existing relaxed structure: {relaxed_pdb_path}")
                pose = pr.pose_from_pdb(relaxed_pdb_path)
            else:
                logger.debug("Starting new relaxation...")
                pose = pr_relax(pose, relaxed_pdb_path)
                if pose is None:
                    logger.warning(
                        "Relaxation failed, falling back to original structure"
                    )
                    pose = pr.pose_from_pdb(pdb_file)
        else:
            logger.debug("Using original structure without relaxation")

        # Analyze interface statistics
        iam = InterfaceAnalyzerMover()
        iam.set_interface("A_B")
        scorefxn = pr.get_fa_scorefxn()
        iam.set_scorefunction(scorefxn)
        iam.set_compute_packstat(True)
        iam.set_compute_interface_energy(True)
        iam.set_calc_dSASA(True)
        iam.set_calc_hbond_sasaE(True)
        iam.set_compute_interface_sc(True)
        iam.set_pack_separated(True)
        iam.apply(pose)

        # Initialize dictionary with all amino acids
        interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}

        # Get interface residues
        interface_residues_set = hotspot_residues(structure_file, binder_chain)
        interface_residues_pdb_ids = []

        # Process interface residues
        for pdb_res_num, aa_type in interface_residues_set.items():
            interface_AA[aa_type] += 1
            interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

        interface_nres = len(interface_residues_pdb_ids)
        interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

        # Calculate interface hydrophobicity
        hydrophobic_aa = set("ACFILMPVWY")
        hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
        interface_hydrophobicity = (
            (hydrophobic_count / interface_nres) * 100 if interface_nres != 0 else 0
        )

        # Get interface statistics
        interfacescore = iam.get_all_data()
        interface_sc = interfacescore.sc_value
        interface_interface_hbonds = interfacescore.interface_hbonds
        interface_dG = iam.get_interface_dG()
        interface_dSASA = iam.get_interface_delta_sasa()
        interface_packstat = iam.get_interface_packstat()
        interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100

        # Calculate unsatisfied H-bonds
        buns_filter = XmlObjects.static_get_filter(
            '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
        )
        interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

        # Calculate H-bond percentages
        if interface_nres != 0:
            interface_hbond_percentage = (
                interface_interface_hbonds / interface_nres
            ) * 100
            interface_bunsch_percentage = (
                interface_delta_unsat_hbonds / interface_nres
            ) * 100
        else:
            interface_hbond_percentage = None
            interface_bunsch_percentage = None

        # Calculate binder energy score
        chain_design = ChainSelector(binder_chain)
        tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
        tem.set_scorefunction(scorefxn)
        tem.set_residue_selector(chain_design)
        binder_score = tem.calculate(pose)

        # Calculate binder SASA
        bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
        bsasa.set_residue_selector(chain_design)
        binder_sasa = bsasa.calculate(pose)
        interface_binder_fraction = (
            (interface_dSASA / binder_sasa) * 100 if binder_sasa > 0 else 0
        )

        # Debug chain processing
        try:
            # Get chain information first
            chain_ids = []
            for i in range(1, pose.num_chains() + 1):
                chain_begin = pose.chain_begin(i)
                chain_id = pose.pdb_info().chain(chain_begin)
                chain_ids.append(chain_id)
                logger.debug(
                    f"Chain {i}: ID={chain_id}, begins at residue {chain_begin}"
                )

            logger.debug(f"Chain IDs found: {chain_ids}")
            logger.debug(f"Requested binder chain: {binder_chain}")

            # Instead of splitting chains, work with selectors
            chain_selector = ChainSelector(binder_chain)
            binder_pose = pose.clone()

            # Verify the chain selection
            if binder_chain in chain_ids:
                logger.debug(f"Found binder chain: {binder_chain}")
            else:
                logger.warning(
                    f"Warning: Chain {binder_chain} not found in {chain_ids}"
                )
                return None, None, None

            logger.debug("Successfully selected chain")

        except Exception as e:
            logger.error(f"Detailed error in chain processing: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback

            traceback.print_exc()
            return None, None, None

        # Calculate surface hydrophobicity
        layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)

        exp_apol_count = 0
        total_count = 0

        for i in range(1, len(surface_res) + 1):
            if surface_res[i]:
                res = binder_pose.residue(i)
                if res.is_apolar() or res.name() in ["PHE", "TRP", "TYR"]:
                    exp_apol_count += 1
                total_count += 1

        surface_hydrophobicity = exp_apol_count / total_count if total_count > 0 else 0

        # Compile interface scores
        interface_scores = {
            "binder_score": binder_score,  # Total energy score of the binder chain
            "surface_hydrophobicity": surface_hydrophobicity,  # Fraction of surface residues that are hydrophobic. Higher values indicate more hydrophobic surface
            "interface_sc": interface_sc,  # Shape complementarity score (0-1). Higher values indicate better geometric fit
            "interface_packstat": interface_packstat,  # Packing statistics score (0-1). Higher values indicate better packing quality
            "interface_dG": interface_dG,  # Binding energy (Rosetta Energy Units). More negative values indicate stronger bindin
            "interface_dSASA": interface_dSASA,  # Change in solvent accessible surface area upon binding. Larger values indicate more surface area buried
            "interface_dG_SASA_ratio": interface_dG_SASA_ratio,  # Binding energy per unit of buried surface. Lower values suggest more efficient binding
            "interface_fraction": interface_binder_fraction,  # Percentage of binder surface involved in interface. Higher values indicate more extensive interface
            "interface_hydrophobicity": interface_hydrophobicity,  # Percentage of interface residues that are hydrophobic. Indicates hydrophobic contribution to binding
            "interface_nres": interface_nres,  # Number of residues in the interface. Indicates interface size
            "interface_interface_hbonds": interface_interface_hbonds,  # Number of hydrogen bonds across interface. More H-bonds suggest stronger specific binding
            "interface_hbond_percentage": interface_hbond_percentage,  # H-bonds per interface residue. Indicates density of H-bond network
            "interface_delta_unsat_hbonds": interface_delta_unsat_hbonds,  # Number of unsatisfied H-bond donors/acceptors. Lower values indicate better H-bond satisfaction
            "interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,  # Unsatisfied H-bonds per interface residue. Lower values indicate better H-bond networks
        }

        # Round float values
        interface_scores = {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in interface_scores.items()
        }

        return interface_scores, interface_AA, interface_residues_pdb_ids_str

    except Exception as e:
        logger.error(f"Error during interface scoring: {str(e)}")
        return None, None, None


def pr_relax(pose, relaxed_pdb_path=None):
    """
    Relax a pose using FastRelax protocol
    Args:
        pose: PyRosetta pose object
        relaxed_pdb_path: Optional path to save relaxed structure
    Returns:
        Relaxed pose or None if relaxation fails
    """
    logger = logging.getLogger(__name__)

    try:
        logger.debug("Starting FastRelax protocol")
        if relaxed_pdb_path:
            logger.debug(f"Relaxed structure will be saved to: {relaxed_pdb_path}")

        start_pose = pose.clone()

        # Generate movemaps
        logger.debug("Configuring MoveMap")
        mmf = pr.rosetta.core.kinematics.MoveMap()
        mmf.set_chi(True)  # enable sidechain movement
        mmf.set_bb(True)  # enable backbone movement
        mmf.set_jump(False)  # disable whole chain movement

        # Run FastRelax
        logger.debug("Setting up FastRelax..")
        fastrelax = pr.rosetta.protocols.relax.FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf)
        fastrelax.max_iter(200)  # default iterations is 2500
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)

        # Apply relaxation
        logger.debug("Running FastRelax optimization")
        fastrelax.apply(pose)
        logger.debug("FastRelax completed successfully")

        # Optionally save relaxed structure
        if relaxed_pdb_path:
            pose.dump_pdb(relaxed_pdb_path)
            logger.debug(f"Saved relaxed structure to {relaxed_pdb_path}")

        return pose

    except Exception as e:
        logger.error(f"FastRelax failed: {str(e)}")
        return None


def score_structure_worker(args: Tuple[str, str, bool]) -> Tuple[str, Dict[str, Any]]:
    """
    Worker function for parallel processing of structure scoring
    
    Args:
        args: Tuple of (structure_file, binder_chain, do_relax)
        
    Returns:
        Tuple of (structure_file, results_dict)
    """
    structure_file, binder_chain, do_relax = args
    
    try:
        scores, aa_comp, interface_residues = score_interface(structure_file, binder_chain, do_relax)
        return structure_file, {
            'scores': scores,
            'aa_comp': aa_comp,
            'interface_residues': interface_residues
        }
    except Exception as e:
        logger.error(f"Error processing {structure_file}: {str(e)}")
        return structure_file, None


def score_structures_parallel(structure_files: List[str], 
                            binder_chain: str = "A", 
                            do_relax: bool = False,
                            max_workers: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Score multiple structures in parallel
    
    Args:
        structure_files: List of structure file paths
        binder_chain: Chain ID of the binder
        do_relax: Whether to perform FastRelax
        max_workers: Maximum number of parallel workers (None for auto)
        
    Returns:
        Dictionary mapping structure files to their results
    """
    logger.info(f"Starting parallel scoring of {len(structure_files)} structures")
    
    # Prepare arguments for workers
    work_items = [(f, binder_chain, do_relax) for f in structure_files]
    
    # Process structures in parallel with progress bar
    results = {}
    
    # Use a more responsive approach with as_completed
    from concurrent.futures import as_completed
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(score_structure_worker, item): item[0] for item in work_items}
        
        # Create progress bar
        with tqdm(total=len(work_items), desc="Scoring interfaces", unit="structure", leave=True) as pbar:
            # Process results as they complete
            for future in as_completed(future_to_file):
                structure_file = future_to_file[future]
                try:
                    result_file, result = future.result()
                    if result is not None:
                        results[result_file] = result
                except Exception as e:
                    logger.error(f"Error processing {structure_file}: {str(e)}")
                
                # Update progress bar
                pbar.update(1)
            
    logger.debug(f"Completed scoring {len(results)} structures")
    return results


def main():
    parser = argparse.ArgumentParser(description="Score protein-protein interface")
    parser.add_argument("structure", help="Input structure file (.cif or .pdb)")
    parser.add_argument("--chain", default="A", help="Binder chain ID (default: B)")
    parser.add_argument(
        "--do-relax",
        action="store_true",
        help="Perform FastRelax before interface scoring",
    )
    args = parser.parse_args()

    # Log at appropriate levels to match plot_metrics.py workflow
    logger.info(f"Starting interface analysis for {args.structure}")
    logger.debug(f"Parameters: chain={args.chain}, do_relax={args.do_relax}")

    # Score interface
    scores, aa_comp, interface_residues = score_interface(
        args.structure, args.chain, args.do_relax
    )

    if scores is None:
        logger.error("Interface scoring failed")
        return

    # Print and log results at appropriate levels
    logger.debug("\nInterface Scores:")
    for key, value in scores.items():
        logger.debug(f"{key}: {value}")

    logger.debug("\nInterface Amino Acid Composition:")
    for aa, count in aa_comp.items():
        if count > 0:
            logger.debug(f"{aa}: {count}")

    logger.debug("\nInterface Residues:")
    logger.debug(f"Interface residues: {interface_residues}")
