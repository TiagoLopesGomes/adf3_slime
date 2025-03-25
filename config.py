import argparse
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class PTMConfig:
    ptmType: str  # e.g., "PTR" for phosphotyrosine
    ptmPosition: int  # Position in the fragment

@dataclass
class ProteinConfig:
    id: str  # Chain ID ("A" for peptide, "B" for receptor)
    sequence: str
    modifications: List[PTMConfig] = field(default_factory=list)

@dataclass
class AF3Config:
    name: str
    modelSeeds: List[int]
    sequences: List[dict]  # Will contain protein configs
    dialect: str = "alphafold3"
    version: int = 2

@dataclass
class Config:
    receptor_path: Path
    ligand_path: Path
    peptide_size: int
    offset: int
    start_residue: int
    ptm_config_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    af3_script_path: Path = Path('~/software/alphafold3/run/run_af3.py')  # Updated default path

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description='AF3 Peptide Fragment Analysis')
        parser.add_argument('--receptor', dest='receptor_path', type=Path, required=True,
                          help='Receptor FASTA file path')
        parser.add_argument('--ligand', dest='ligand_path', type=Path, required=True,
                          help='Ligand FASTA file path')
        parser.add_argument('--peptide-size', dest='peptide_size', type=int, default=10,
                          help='Size of peptide fragments')
        parser.add_argument('--offset', dest='offset', type=int, default=3,
                          help='Offset between fragments')
        parser.add_argument('--start-residue', dest='start_residue', type=int, required=True,
                          help='Start residue number in full sequence')
        parser.add_argument('--ptm-config', dest='ptm_config_path', type=Path,
                          help='PTM configuration YAML file')
        parser.add_argument('--output-dir', dest='output_dir', type=Path,
                          help='Output directory')
        parser.add_argument('--af3-script', dest='af3_script_path', type=Path,
                          default='~/software/alphafold3/run/run_af3.py',
                          help='Path to AF3 run script')
        
        args = parser.parse_args()
        # Expand any user paths (~ or ~user)
        if args.af3_script_path:
            args.af3_script_path = args.af3_script_path.expanduser()
        if args.output_dir:
            args.output_dir = args.output_dir.expanduser()
            
        return cls(**vars(args))

    def create_af3_config(self, fragment_name: str, peptide_seq: str, receptor_seq: str, 
                         fragment_start: int, ptms: List[PTMConfig] = None) -> AF3Config:
        """Create AF3 configuration for a specific fragment"""
        
        # Create protein configs
        peptide_config = {
            "protein": {
                "id": "A",
                "sequence": peptide_seq,
            }
        }
        
        # Add modifications if present
        if ptms:
            peptide_config["protein"]["modifications"] = [
                {"ptmType": ptm.ptmType, "ptmPosition": ptm.ptmPosition}
                for ptm in ptms
            ]

        receptor_config = {
            "protein": {
                "id": "B",
                "sequence": receptor_seq
            }
        }

        return AF3Config(
            name=f"{fragment_name}_{fragment_start}",
            modelSeeds=[1],  # Default value, could be made configurable
            sequences=[peptide_config, receptor_config]
        )

    def save_af3_config(self, config: AF3Config, output_path: Path) -> None:
        """Save AF3 configuration to JSON file"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(vars(config), f, indent=2)

    def validate(self):
        """Basic validation of configuration"""
        if not self.receptor_path.exists():
            raise ValueError(f"Receptor file not found: {self.receptor_path}")
        if not self.ligand_path.exists():
            raise ValueError(f"Ligand file not found: {self.ligand_path}")
        if self.peptide_size < 1:
            raise ValueError(f"Invalid peptide size: {self.peptide_size}")
        if self.offset < 1:
            raise ValueError(f"Invalid offset: {self.offset}")
        if self.ptm_config_path and not self.ptm_config_path.exists():
            raise ValueError(f"PTM config file not found: {self.ptm_config_path}")

    def load_ptm_config(self) -> List[PTMConfig]:
        """Load PTM configuration from YAML file"""
        if not self.ptm_config_path:
            return []
            
        with open(self.ptm_config_path) as f:
            ptm_data = yaml.safe_load(f)
            
        return [PTMConfig(**ptm) for ptm in ptm_data.get('modifications', [])]
