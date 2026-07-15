import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class TrainLoggingTests(unittest.TestCase):
    def test_setup_train_logging_splits_named_log_files(self):
        from lib.train.admin.logging_utils import setup_train_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SimpleNamespace(
                script_name="seatrack",
                config_name="rgbt",
                local_rank=0,
                save_dir=tmpdir,
            )

            loggers = setup_train_logging(settings)
            loggers["train"].info("train message")
            loggers["system"].warning("system message")
            loggers["config"].info("config message")
            loggers["model"].info("model message")
            loggers["data"].info("data message")

            for logger in loggers.values():
                for handler in logger.handlers:
                    handler.flush()

            log_dir = Path(tmpdir) / "logs"
            expected_files = {
                "train": log_dir / "seatrack-rgbt.train.log",
                "system": log_dir / "seatrack-rgbt.system.log",
                "config": log_dir / "seatrack-rgbt.config.log",
                "model": log_dir / "seatrack-rgbt.model.log",
                "data": log_dir / "seatrack-rgbt.data.log",
            }

            for name, path in expected_files.items():
                self.assertTrue(path.exists(), f"{name} log file was not created")

            self.assertIn("train message", expected_files["train"].read_text())
            self.assertNotIn("system message", expected_files["train"].read_text())
            self.assertIn("system message", expected_files["system"].read_text())
            self.assertIn("config message", expected_files["config"].read_text())
            self.assertIn("model message", expected_files["model"].read_text())
            self.assertIn("data message", expected_files["data"].read_text())
            self.assertEqual(str(expected_files["train"]), settings.log_file)

    def test_setup_train_logging_routes_module_loggers(self):
        from lib.train.admin.logging_utils import setup_train_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SimpleNamespace(
                script_name="seatrack",
                config_name="rgbt",
                local_rank=0,
                save_dir=tmpdir,
            )

            setup_train_logging(settings)
            logging.getLogger("lib.models.seatrack.vit").info("model namespace message")
            logging.getLogger("lib.train.data.image_loader").error("data namespace message")

            for logger_name in ("lib.models", "lib.train.data"):
                for handler in logging.getLogger(logger_name).handlers:
                    handler.flush()

            log_dir = Path(tmpdir) / "logs"
            self.assertIn("model namespace message", (log_dir / "seatrack-rgbt.model.log").read_text())
            self.assertIn("data namespace message", (log_dir / "seatrack-rgbt.data.log").read_text())

    def test_get_train_logger_returns_null_logger_when_not_configured(self):
        from lib.train.admin.logging_utils import get_train_logger

        logger = get_train_logger(SimpleNamespace(), "train")
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)
        self.assertFalse(logger.propagate)


if __name__ == "__main__":
    unittest.main()
