import os
import sys
import logging
from logging import handlers
from pathlib import Path


def prepare_log(base_dir, filename="info"):
    fmt = '%(asctime)s [%(levelname)s: %(filename)s %(lineno)d] %(message)s'
    log_dir = Path(base_dir) / "logs"
    print(log_dir)

    log_dir.mkdir(exist_ok=True)
    logs = log_dir.glob(filename+"*")
    # for log_fname in logs:
    #     os.remove(log_fname)

    log_path = log_dir / f"{filename}.log"
    return fmt, log_path


def set_log(level=logging.INFO, filename="info", base_dir=r"."):
    fmt, log_path = prepare_log(base_dir, filename)
    fhandler = handlers.RotatingFileHandler(log_path, maxBytes=100000000, backupCount=5)
    logging.basicConfig(level=level, format=fmt,force=True,
                        handlers=[fhandler, logging.StreamHandler(stream=sys.stdout)])
    logging.info("path: {}".format(log_path))
