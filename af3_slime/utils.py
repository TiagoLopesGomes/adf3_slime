import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir: Path, timestamp: str = None) -> logging.Logger:
    """Setup logging to both file and console with different levels
    
    Args:
        output_dir: Directory where log file will be created
        timestamp: Optional timestamp for log filename. If None, current time will be used
        
    Returns:
        Logger instance configured with file and console handlers
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log filename with timestamp
    log_file = output_dir / f"{timestamp}.log"

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler - only INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # File handler - DEBUG and above
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Log file created at: {log_file}")
    
    return logger 