import logging
import os
from functools import wraps
import colorlog
from datetime import datetime

def setup_logger(name, log_dir='/home/ashish/Desktop/202418007/MeanReversion_usingPaper/logs'):
    """
    Creates and returns a logger with both color console and file output.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name}.log')

    # colored console logs
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Stream Handler (Console)
    stream_handler = colorlog.StreamHandler()
    stream_handler.setFormatter(color_formatter)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Logger setup
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent duplicate handlers
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger

# Logging decorator
def log_function_call(get_logger):
    """
    Decorator to log function entry and exit using the provided logger getter.
    Use as: @log_function_call(lambda self: self.logger)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger if callable(get_logger) else None
            if logger:
                logger.info(f"Calling function {func.__name__}")
            result = func(*args, **kwargs)
            if logger:
                logger.info(f"Function {func.__name__} completed")
            return result
        return wrapper
    return decorator