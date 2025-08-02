import logging
import sys
import os
from typing import Optional

def setup_logger(log_file: str, level: int = logging.INFO, name: Optional[str] = None):
    log_level_name = logging.getLevelName(level)
    # Configure the root logger if no specific name is given, or the named logger
    logger_to_configure = logging.getLogger(name) # name=None gets the root logger
    logger_to_configure.setLevel(level)

    # Remove existing handlers to prevent duplication if this function is called multiple times
    # especially important for the root logger.
    if logger_to_configure.hasHandlers():
        for handler in logger_to_configure.handlers[:]: # Iterate over a copy
            logger_to_configure.removeHandler(handler)
            handler.close() # Close the handler properly
            
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True) # Ensure log directory exists
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level) # Handler level should also be set

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level) # Handler level should also be set

    logger_to_configure.addHandler(file_handler)
    logger_to_configure.addHandler(stream_handler)

    # If we configured a specific logger (not root), we usually want it to propagate to root 
    # unless specific non-propagation is desired. Root logger's propagate is False by default.
    if name:
        logger_to_configure.propagate = True # Default is True anyway, but explicit for named loggers
    
    # This function now primarily configures, doesn't need to return the logger instance
    # The caller can get the logger via logging.getLogger(name_they_want)
    # Inform that logger is configured
    # Use a temporary logger for this message to avoid issues if the logger being configured is complex
    temp_logger = logging.getLogger("LoggerSetup")
    if not temp_logger.hasHandlers(): # Basic config for this temp logger if none
        temp_stream_handler = logging.StreamHandler(sys.stdout)
        temp_stream_handler.setFormatter(formatter)
        temp_logger.addHandler(temp_stream_handler)
        temp_logger.setLevel(logging.INFO)
    temp_logger.info(f"Logger '{name if name else 'root'}' configured. Level: {log_level_name}. File: {log_file}")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)





