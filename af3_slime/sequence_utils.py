import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from Bio import SeqIO
from config import PTMConfig
from ptm_resi import CCD_NAME_TO_ONE_LETTER

logger = logging.getLogger(__name__)

@dataclass
class Fragment:
    sequence: str
    start: int
    end: int
    ptms: List[PTMConfig] = None

class SequenceProcessor:
    def __init__(self, receptor_path: Path, ligand_path: Path):
        self.receptor_path = receptor_path
        self.ligand_path = ligand_path
        logger.debug("SequenceProcessor Initialization:")
        logger.info(f"Reading receptor from: {receptor_path}")
        logger.info(f"Reading ligand from: {ligand_path}")
        self.receptor_name, self.receptor_seq = self._read_fasta(receptor_path)
        self.ligand_name, self.ligand_seq = self._read_fasta(ligand_path)
        logger.info(f"Loaded receptor: {self.receptor_name}, length: {len(self.receptor_seq)}")
        logger.info(f"Loaded ligand: {self.ligand_name}, length: {len(self.ligand_seq)}")
        print("\n")

    def _read_fasta(self, file_path: Path) -> Tuple[str, str]:
        """Read FASTA file and return name and sequence"""
        with open(file_path, 'r') as file:
            record = next(SeqIO.parse(file, 'fasta'))
            return record.id, str(record.seq)

    def generate_fragments(self, peptide_size: int, offset: int, start_residue: int,
                         global_ptms: List[PTMConfig] = None) -> List[Fragment]:
        """Generate peptide fragments with PTMs in absolute positions"""
        fragments = []
        sequence_length = len(self.ligand_seq)
        i = 0

        while i + peptide_size <= sequence_length:
            fragment_start = i + start_residue
            fragment_end = fragment_start + peptide_size - 1
            fragment_seq = self.ligand_seq[i:i + peptide_size]
            
            # Map global PTMs to fragment positions
            fragment_ptms = []
            if global_ptms:
                for ptm in global_ptms:
                    if fragment_start <= ptm.ptmPosition <= fragment_end:
                        # Convert to relative position within fragment
                        rel_pos = ptm.ptmPosition - fragment_start + 1
                        fragment_ptms.append(PTMConfig(
                            ptmType=ptm.ptmType,
                            ptmPosition=rel_pos
                        ))

            fragments.append(Fragment(
                sequence=fragment_seq,
                start=fragment_start,
                end=fragment_end,
                ptms=fragment_ptms
            ))
            i += offset

        return fragments

    def validate_ptms(self, ptms: List[PTMConfig], start_residue: int) -> bool:
        """
        Validate PTM positions against the full ligand sequence
        Takes into account the absolute position offset and validates PTM types
        against allowed modifications from CCD_NAME_TO_ONE_LETTER dictionary
        """
        ligand_length = len(self.ligand_seq)
        end_residue = start_residue + ligand_length - 1
        
        logger.debug("PTM Validation Debug:")
        logger.debug(f"Ligand sequence length: {ligand_length}")
        logger.debug(f"Absolute ligand position range: {start_residue}-{end_residue}")
        logger.debug(f"Ligand sequence: {self.ligand_seq}")
        logger.info(f"Checking PTM positions: {[ptm.ptmPosition for ptm in ptms]}")
        
        for ptm in ptms:
            logger.info(f"Checking PTM at absolute position {ptm.ptmPosition}")
            
            # Check if PTM type exists in the dictionary
            if ptm.ptmType not in CCD_NAME_TO_ONE_LETTER:
                logger.error(f"Error: PTM type '{ptm.ptmType}' is not a valid modification type")
                return False
                
            # Get the expected residue type for this PTM
            expected_residue = CCD_NAME_TO_ONE_LETTER[ptm.ptmType]
            logger.debug(f"PTM type '{ptm.ptmType}' expects residue '{expected_residue}'")
            
            # Convert absolute position to sequence position
            seq_position = ptm.ptmPosition - start_residue + 1
            logger.debug(f"Converted to sequence position: {seq_position}")
            
            # Check if PTM position exists in the sequence
            if seq_position < 1 or seq_position > ligand_length:
                logger.error(f"PTM position {ptm.ptmPosition} (seq pos {seq_position}) is outside sequence range")
                return False
            
            # Check if the residue matches the expected type for this PTM
            actual_residue = self.ligand_seq[seq_position-1]
            if actual_residue != expected_residue:
                logger.error(f"Error: Position {ptm.ptmPosition} has residue '{actual_residue}' but PTM type '{ptm.ptmType}' requires '{expected_residue}'")
                return False
            
            logger.info(f"PTM validation successful: Position {ptm.ptmPosition} has correct residue '{actual_residue}' for PTM type '{ptm.ptmType}'")
            
        return True

    def convert_to_relative_position(self, abs_position: int, fragment_start: int) -> int:
        """Convert absolute PTM position to relative position within fragment"""
        return abs_position - fragment_start + 1

    def get_receptor_info(self) -> Dict[str, str]:
        """Get receptor information for AF3 config"""
        return {
            "name": self.receptor_name,
            "sequence": self.receptor_seq
        }

    def get_fragment_name(self, fragment: Fragment) -> str:
        """Generate standardized name for fragment"""
        return f"{self.receptor_name}_{self.ligand_name}_{fragment.start}_{fragment.end}"
