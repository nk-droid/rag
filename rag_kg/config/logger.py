import os
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)

        # Formatter
        formatter = logging.Formatter(
            '%(levelname)s: %(name)s: %(message)s'
        )
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)
        logger.propagate = False

    return logger