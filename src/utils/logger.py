import os
import datetime
import logging


def set_file_logger(name: str = None) -> logging.Logger:

    log_format = '[%(levelname)s] %(asctime)s - %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    log_level = logging.DEBUG

    tz = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=tz)

    log_file_path = os.path.join(os.path.dirname(__file__), '../logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(log_format, log_datefmt)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = set_file_logger("utils.logger")