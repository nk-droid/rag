import logging

from infra.logging.recent_logs import RecentLogsHandler

_LOGGERS = {}
_CONFIGURED = False

def get_logger(name: str, level: str = "INFO"):
    global _CONFIGURED
    if name in _LOGGERS:
        return _LOGGERS[name]

    if not _CONFIGURED:
        handler = RecentLogsHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[handler],
        ) 
        _CONFIGURED = True

    logger = logging.getLogger(name)
    logger.setLevel(level)

    _LOGGERS[name] = logger
    return logger
