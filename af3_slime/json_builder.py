import json
from pathlib import Path
from typing import List

from sequence_utils import Fragment


class JsonBuilder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_fragment_json(self, 
                           fragment: Fragment,
                           receptor_name: str,
                           receptor_seq: str,
                           ligand_name: str) -> Path:
        """
        Creates AF3 input JSON for a specific fragment
        Returns the path to the created JSON file
        """
        # Create the fragment name
        fragment_name = f"{receptor_name}_{ligand_name}_{fragment.start}_{fragment.end}"
        
        # Create protein configs
        peptide_config = {
            "protein": {
                "id": "A",
                "sequence": fragment.sequence
            }
        }

        # Add modifications if present
        if fragment.ptms:
            peptide_config["protein"]["modifications"] = [
                {"ptmType": ptm.ptmType, "ptmPosition": ptm.ptmPosition}
                for ptm in fragment.ptms
            ]

        receptor_config = {
            "protein": {
                "id": "B",
                "sequence": receptor_seq
            }
        }

        # Create the full AF3 config
        af3_config = {
            "name": fragment_name,
            "modelSeeds": [1],
            "sequences": [peptide_config, receptor_config],
            "dialect": "alphafold3",
            "version": 2
        }

        # Create output file path
        json_path = self.output_dir / f"{fragment_name}.json"
        
        # Write the JSON file
        with open(json_path, 'w') as f:
            json.dump(af3_config, f, indent=2)

        return json_path

    def create_all_fragment_jsons(self,
                                fragments: List[Fragment],
                                receptor_name: str,
                                receptor_seq: str,
                                ligand_name: str) -> List[Path]:
        """
        Creates AF3 input JSONs for all fragments
        Returns list of paths to created JSON files
        """
        json_paths = []
        for fragment in fragments:
            json_path = self.create_fragment_json(
                fragment=fragment,
                receptor_name=receptor_name,
                receptor_seq=receptor_seq,
                ligand_name=ligand_name
            )
            json_paths.append(json_path)
        return json_paths

    def validate_json(self, json_path: Path) -> bool:
        """
        Validates that a created JSON file matches AF3 requirements
        """
        try:
            with open(json_path) as f:
                data = json.load(f)
                
            # Basic structure validation
            required_keys = {"name", "modelSeeds", "sequences", "dialect", "version"}
            if not all(key in data for key in required_keys):
                return False
                
            # Validate sequences
            if not isinstance(data["sequences"], list) or len(data["sequences"]) != 2:
                return False
                
            # Validate each protein configuration
            for seq in data["sequences"]:
                if "protein" not in seq:
                    return False
                protein = seq["protein"]
                if not all(key in protein for key in ["id", "sequence"]):
                    return False
                    
            return True
            
        except (json.JSONDecodeError, IOError):
            return False
