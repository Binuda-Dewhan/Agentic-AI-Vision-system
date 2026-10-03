import logging


def setup_logging(level=logging.INFO, verbose=False):
    """Setup standard logging for the application."""
    if verbose:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("sleep_monitor")
