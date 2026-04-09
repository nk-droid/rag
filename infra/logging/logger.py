import logging
from rich.logging import RichHandler

_LOGGERS = {}

def get_logger(name: str, level: str = "INFO"):
    if name in _LOGGERS:
        return _LOGGERS[name]

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False
            )
        ]
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    _LOGGERS[name] = logger
    return logger