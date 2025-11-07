import logging
import sys

def configure_logging(level=logging.INFO):
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    for noisy in ("urllib3", "rdflib", "asyncio", "uvicorn.error"):
        logging.getLogger(noisy).setLevel(logging.WARNING)