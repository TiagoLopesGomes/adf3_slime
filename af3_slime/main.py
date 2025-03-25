import logging
from pathlib import Path
from config import Config
from sequence_utils import SequenceProcessor
from json_builder import JsonBuilder
from af3_runner import AF3Runner
from datetime import datetime
from utils import setup_logging

def print_banner():
    banner = """


    █████╗ ███████╗  ▃▃▃▃▃▃       ███████╗██╗     ██╗███╗   ███╗███████╗ 
   ██╔══██╗██╔════╝   ▃▃▃▃▃▃      ██╔════╝██║     ██║████╗ ████║██╔════╝   
   ███████║█████╗      ▃▃▃▃▃▃     ███████╗██║     ██║██╔████╔██║█████╗       
   ██╔══██║██╔══╝       ▃▃▃▃▃▃    ╚════██║██║     ██║██║╚██╔╝██║██╔══╝         
   ██║  ██║██║           ▃▃▃▃▃▃   ███████║███████╗██║██║ ╚═╝ ██║███████╗        
   ╚═╝  ╚═╝╚═╝                    ╚══════╝╚══════╝╚═╝╚═╝     ╚═╝╚══════╝         
                              
          [ AlphaFold3 SLIM discovery pipeline ]
          

"""
    print(banner)

def main():
    # Record start time
    start_time = datetime.now()
    
    print_banner()  # Print banner at start
    
    # Load configuration from command line arguments
    config = Config.from_args()
    config.validate()
    
    # Create output directory if specified, otherwise use default
    if not config.output_dir:
        config.output_dir = Path("af3_predictions")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory inside output directory
    logs_dir = config.output_dir / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging using shared setup
    logger = setup_logging(logs_dir, timestamp)
    
    logger.info("Starting AF3 peptide fragment analysis pipeline")
    
    try:
        # Initialize sequence processor
        processor = SequenceProcessor(config.receptor_path, config.ligand_path)
        
        # Load PTM configurations if provided
        ptms = config.load_ptm_config()
        if ptms:
            logger.info(f"Loaded {len(ptms)} PTM configurations")
            if not processor.validate_ptms(ptms, config.start_residue):
                raise ValueError("Invalid PTM positions detected")
        
        # Generate fragments
        fragments = processor.generate_fragments(
            peptide_size=config.peptide_size,
            offset=config.offset,
            start_residue=config.start_residue,
            global_ptms=ptms
        )
        print("\n")
        logger.info(f"Generated {len(fragments)} peptide fragments")
        
        # Create JSON files for each fragment
        json_builder = JsonBuilder(config.output_dir)
        json_paths = json_builder.create_all_fragment_jsons(
            fragments=fragments,
            receptor_name=processor.receptor_name,
            receptor_seq=processor.receptor_seq,
            ligand_name=processor.ligand_name
        )
        logger.info(f"Created {len(json_paths)} AF3 input JSON files")
        
        # Validate JSON files
        for json_path in json_paths:
            if not json_builder.validate_json(json_path):
                raise ValueError(f"Invalid JSON created: {json_path}")
        
        # Initialize and run AF3 predictions
        runner = AF3Runner(
            output_base_dir=config.output_dir,
            af3_script_path=config.af3_script_path
        )
        
        # Run predictions
        output_dirs = runner.run_all_predictions(json_paths)
        logger.info(f"Completed {len(output_dirs)} AF3 predictions")
        
        # Verify outputs without cleanup
        '''failed_runs = []
        for output_dir in output_dirs:
            if not runner.check_output(output_dir):
                failed_runs.append(output_dir)
                # Skip cleanup call
                logger.warning(f"Failed prediction detected: {output_dir}")
        
        if failed_runs:
            logger.warning(f"Failed runs: {len(failed_runs)}")'''
        
        # Calculate total execution time
        execution_time = datetime.now() - start_time
        hours, remainder = divmod(execution_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Total execution time: {time_str}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()