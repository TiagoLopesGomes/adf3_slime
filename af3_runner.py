import os
import subprocess
from pathlib import Path
from typing import List
import logging
from datetime import datetime

class AF3Runner:
    def __init__(self, output_base_dir: Path, af3_script_path: Path):
        """
        Initialize AF3Runner
        Args:
            output_base_dir: Base directory for AF3 outputs
            af3_script_path: Path to AF3 run script
        """
        self.output_base_dir = output_base_dir
        self.af3_script_path = af3_script_path
        self.logger = logging.getLogger(__name__)

    def run_prediction(self, json_path: Path) -> Path:
        """
        Run AF3 prediction for a single JSON configuration
        Returns the path to the output directory
        """
        output_dir = self.output_base_dir / json_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Explicitly expand the home directory
        af3_script = os.path.expanduser("~/software/alphafold3/run/run_af3.py")

        cmd = [
            'python',
            af3_script,  # Now this will be the fully expanded path
            '--json_path', str(json_path),
            '--output_dir', str(output_dir),
            '--run_mmseqs'
        ]

        self.logger.info(f"Starting AF3 prediction for {json_path.name}")
        try:
            subprocess.run(cmd, 
                         check=True,
                         capture_output=True,
                         text=True)
            self.logger.info(f"Completed AF3 prediction for {json_path}")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"AF3 prediction failed for {json_path}")
            self.logger.error(f"Error output: {e.stderr}")
            raise

    def check_output(self, output_dir: Path) -> bool:
        """
        Verify that AF3 prediction output exists and is valid
        """
        # First check if directory exists
        if not output_dir.exists():
            self.logger.debug(f"Output directory does not exist: {output_dir}")
            return False
        
        # Get all files in the directory (lowercase for comparison)
        existing_files = [f.name.lower() for f in output_dir.iterdir() if f.is_file()]
        self.logger.debug(f"Found {len(existing_files)} files in {output_dir}")
        
        # Check for required file patterns
        required_patterns = [
            "_model.cif",
            "_confidences.json",
            "_data.json",
            "_summary_confidences.json",
            "terms_of_use.md",
            "ranking_scores.csv"
        ]
        
        # Check if each required pattern exists in any file (case-insensitive)
        for pattern in required_patterns:
            pattern_lower = pattern.lower()
            if not any(pattern_lower in f for f in existing_files):
                self.logger.debug(f"Missing required file with pattern '{pattern}' in {output_dir}")
                return False
        
        self.logger.debug(f"All required files found in {output_dir}")
        return True

    def cleanup_failed_runs(self, output_dir: Path):
        """
        Clean up incomplete or failed prediction outputs
        Currently disabled - only logs warning
        """
        if output_dir.exists() and not self.check_output(output_dir):
            self.logger.warning(f"Failed run detected in {output_dir}")
            # Skip cleanup for now
            pass

    def run_all_predictions(self, json_paths: List[Path]) -> List[Path]:
        """
        Run AF3 predictions for multiple JSON configurations
        Returns list of output directories
        """
        output_dirs = []
        for json_path in json_paths:
            try:
                output_dir = self.run_prediction(json_path)
                output_dirs.append(output_dir)
            except Exception as e:
                self.logger.error(f"Failed to process {json_path}: {str(e)}")
                continue
        return output_dirs
