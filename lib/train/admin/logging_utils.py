import logging
import os


LOG_CATEGORIES = ("train", "system", "config", "model", "data")
LOG_NAMESPACE_CATEGORIES = {
    "lib.models": "model",
    "lib.train.data": "data",
    "lib.train.dataset": "data",
    "lib.train.admin": "system",
}
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _is_main_process(settings):
    return getattr(settings, "local_rank", -1) in (-1, 0)


def _clear_handlers(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _make_logger(name, handler):
    logger = logging.getLogger(name)
    _clear_handlers(logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger


def _make_file_handler(file_name):
    handler = logging.FileHandler(file_name, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def _make_handler(settings, file_name):
    if _is_main_process(settings):
        return _make_file_handler(file_name)
    return logging.NullHandler()


def _configure_namespace_loggers(settings, log_dir, base_name):
    for namespace, category in LOG_NAMESPACE_CATEGORIES.items():
        file_name = os.path.join(log_dir, f"{base_name}.{category}.log")
        _make_logger(namespace, _make_handler(settings, file_name))


def setup_train_logging(settings, log_dir=None):
    """Create file loggers for a training run and attach them to settings."""
    log_dir = log_dir or os.path.join(settings.save_dir, "logs")
    script_name = getattr(settings, "script_name", "train")
    config_name = getattr(settings, "config_name", "default")
    base_name = f"{script_name}-{config_name}"
    train_log_file = os.path.join(log_dir, f"{base_name}.train.log")

    if _is_main_process(settings):
        os.makedirs(log_dir, exist_ok=True)

    loggers = {}
    for category in LOG_CATEGORIES:
        logger_name = f"seatrack.{base_name}.{category}"
        file_name = os.path.join(log_dir, f"{base_name}.{category}.log")
        handler = _make_handler(settings, file_name)
        loggers[category] = _make_logger(logger_name, handler)

    _configure_namespace_loggers(settings, log_dir, base_name)

    settings.log_dir = log_dir
    settings.log_file = train_log_file
    settings.loggers = loggers
    return loggers


def get_train_logger(settings, category):
    """Return a configured training logger, or a quiet fallback logger."""
    loggers = getattr(settings, "loggers", None)
    if loggers and category in loggers:
        return loggers[category]

    logger = logging.getLogger(f"seatrack.null.{category}")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
