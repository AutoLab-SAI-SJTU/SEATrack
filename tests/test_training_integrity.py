import unittest
from copy import deepcopy
import importlib
import random
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import cv2 as cv
import numpy as np
import torch

from lib.train.actors.seatrack import SEATrackActor
from lib.config.seatrack.config import cfg as default_cfg
from lib.config.seatrack.config import update_config_from_file
from lib.train.base_functions import build_dataloaders, update_settings
from lib.train.data.loader import LTRLoader
from lib.train.data.transforms import ToGrayscale
from lib.train.trainers.ltr_trainer import LTRTrainer
from tracking.train import _append_common_train_args


class TrainingIntegrityTests(unittest.TestCase):
    @staticmethod
    def _loader_order(loader):
        return [index for batch in loader for index in batch[0].tolist()]

    def test_loader_shuffle_is_isolated_from_global_model_rng(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(24))

        first = LTRLoader(
            "train",
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(17),
        )
        torch.rand(10)
        first_order = self._loader_order(first)

        second = LTRLoader(
            "train",
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(17),
        )
        torch.rand(1000)
        second_order = self._loader_order(second)

        self.assertEqual(first_order, second_order)

    def test_loader_epoch_seed_is_recoverable(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(24))

        def order_for_epoch(epoch):
            loader = LTRLoader(
                "train",
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                base_seed=31,
            )
            loader.set_epoch(epoch)
            return self._loader_order(loader)

        self.assertEqual(order_for_epoch(3), order_for_epoch(3))
        self.assertNotEqual(order_for_epoch(3), order_for_epoch(4))

    def test_trainer_reseeds_loader_before_each_epoch_cycle(self):
        events = []

        class FakeLoader:
            epoch_interval = 1
            sampler = None

            def set_epoch(self, epoch):
                events.append(("set_epoch", epoch))

        trainer = object.__new__(LTRTrainer)
        trainer.epoch = 4
        trainer.loaders = [FakeLoader()]
        trainer.settings = SimpleNamespace(local_rank=1)
        trainer.cycle_dataset = lambda loader: events.append(("cycle", trainer.epoch))
        trainer._stats_new_epoch = lambda: None

        trainer.train_epoch()

        self.assertEqual(events, [("set_epoch", 4), ("cycle", 4)])

    def test_build_dataloaders_assigns_separate_recoverable_stream_seeds(self):
        cfg = deepcopy(default_cfg)
        update_config_from_file("experiments/seatrack/rgbt_lifttrack_pilot.yaml", cfg)
        settings = SimpleNamespace(local_rank=-1, seed=7, use_lmdb=False)
        update_settings(settings, cfg)
        dummy_train = torch.utils.data.TensorDataset(torch.arange(64))
        dummy_val = torch.utils.data.TensorDataset(torch.arange(64))

        with patch(
            "lib.train.base_functions.names2datasets",
            return_value=[object()],
        ), patch(
            "lib.train.base_functions.sampler.TrackingSampler",
            side_effect=[dummy_train, dummy_val],
        ):
            train_loader, val_loader = build_dataloaders(cfg, settings)

        self.assertEqual(train_loader.base_seed, 7)
        self.assertEqual(val_loader.base_seed, 1_000_007)
        self.assertIsNotNone(train_loader.worker_init_fn)
        self.assertIsNotNone(val_loader.worker_init_fn)

    def test_training_entry_propagates_cli_seed_to_settings(self):
        with patch.dict("sys.modules", {"_init_paths": SimpleNamespace()}):
            training_entry = importlib.import_module("lib.train.run_training")
        settings = SimpleNamespace()
        observed = {}
        expression = SimpleNamespace(run=lambda current: observed.setdefault("settings", current))

        with patch.object(training_entry.ws_settings, "Settings", return_value=settings), patch.object(
            training_entry.importlib,
            "import_module",
            return_value=expression,
        ), patch.object(training_entry, "init_seeds"):
            training_entry.run_training(
                "seatrack",
                "rgbt_lifttrack_pilot",
                save_dir="/tmp/lifttrack-seed-test",
                base_seed=9,
            )

        self.assertIs(observed["settings"], settings)
        self.assertEqual(settings.seed, 9)

    def test_training_wrapper_forwards_seed_to_subprocess(self):
        args = SimpleNamespace(
            script="seatrack",
            config="rgbt_lifttrack_pilot",
            save_dir="/tmp/lifttrack-seed-test",
            seed=7,
            use_lmdb=0,
            use_wandb=0,
            distill=0,
            script_prv=None,
            config_prv=None,
            script_teacher=None,
            config_teacher=None,
        )

        command = _append_common_train_args(["python"], args)

        seed_index = command.index("--seed")
        self.assertEqual(command[seed_index + 1], "7")

    def test_rng_state_round_trip_replays_all_generators(self):
        from lib.train.trainers.base_trainer import capture_rng_state, restore_rng_state

        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(19)
        state = capture_rng_state()
        expected = {
            "python": random.random(),
            "numpy": np.random.rand(),
            "torch": torch.rand(4),
        }
        if torch.cuda.is_available():
            expected["cuda"] = torch.rand(4, device="cuda")

        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(91)
        restore_rng_state(state)

        self.assertEqual(random.random(), expected["python"])
        self.assertEqual(np.random.rand(), expected["numpy"])
        self.assertTrue(torch.equal(torch.rand(4), expected["torch"]))
        if torch.cuda.is_available():
            self.assertTrue(torch.equal(torch.rand(4, device="cuda"), expected["cuda"]))

    def test_checkpoint_restores_rng_after_model_reconstruction(self):
        from lib.train.trainers.base_trainer import BaseTrainer

        def make_trainer(root):
            trainer = object.__new__(BaseTrainer)
            net = torch.nn.Linear(3, 2)
            trainer.actor = SimpleNamespace(net=net)
            trainer.optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
            trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
            trainer.settings = SimpleNamespace(project_path="rng-test", local_rank=0)
            trainer._checkpoint_dir = root
            trainer.epoch = 2
            trainer.stats = {"marker": 1}
            trainer.loaders = []
            return trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            random.seed(23)
            np.random.seed(23)
            torch.manual_seed(23)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(23)
            original = make_trainer(tmpdir)
            original.save_checkpoint()
            checkpoint_path = (
                f"{tmpdir}/rng-test/Linear_ep0002.pth.tar"
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertIn("rng_state", checkpoint)
            expected = {
                "python": random.random(),
                "numpy": np.random.rand(),
                "torch": torch.rand(4),
            }
            if torch.cuda.is_available():
                expected["cuda"] = torch.rand(4, device="cuda")

            resumed = make_trainer(tmpdir)
            random.seed(88)
            np.random.seed(88)
            torch.manual_seed(88)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(88)
            self.assertTrue(resumed.load_checkpoint())

            self.assertEqual(resumed.epoch, 2)
            self.assertEqual(random.random(), expected["python"])
            self.assertEqual(np.random.rand(), expected["numpy"])
            self.assertTrue(torch.equal(torch.rand(4), expected["torch"]))
            if torch.cuda.is_available():
                self.assertTrue(torch.equal(torch.rand(4, device="cuda"), expected["cuda"]))

    def test_grayscale_converts_each_modality_without_dropping_channels(self):
        image = np.zeros((2, 2, 6), dtype=np.uint8)
        image[..., :3] = np.array([10, 20, 30], dtype=np.uint8)
        image[..., 3:] = np.array([90, 60, 30], dtype=np.uint8)

        result = ToGrayscale(probability=1.0).transform_image(image, True)

        self.assertEqual(result.shape, image.shape)
        expected_first = cv.cvtColor(image[..., :3], cv.COLOR_RGB2GRAY)
        expected_second = cv.cvtColor(image[..., 3:], cv.COLOR_RGB2GRAY)
        for channel in range(3):
            np.testing.assert_array_equal(result[..., channel], expected_first)
        for channel in range(3, 6):
            np.testing.assert_array_equal(result[..., channel], expected_second)
        self.assertFalse(np.array_equal(result[..., 0], result[..., 3]))

    def test_grayscale_preserves_three_channel_compatibility(self):
        image = np.array(
            [
                [[10, 20, 30], [30, 20, 10]],
                [[90, 60, 30], [5, 100, 200]],
            ],
            dtype=np.uint8,
        )

        result = ToGrayscale(probability=1.0).transform_image(image, True)

        expected = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        self.assertEqual(result.shape, image.shape)
        for channel in range(3):
            np.testing.assert_array_equal(result[..., channel], expected)

    def test_grayscale_rejects_unsupported_dimensions_and_channel_counts(self):
        invalid_images = {
            "two-dimensional": np.zeros((2, 2), dtype=np.uint8),
            "four-dimensional": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "one-channel": np.zeros((2, 2, 1), dtype=np.uint8),
            "four-channel": np.zeros((2, 2, 4), dtype=np.uint8),
            "nine-channel": np.zeros((2, 2, 9), dtype=np.uint8),
        }

        transform = ToGrayscale(probability=1.0)
        for layout, image in invalid_images.items():
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    ValueError,
                    "ToGrayscale expects an HxWx3 or HxWx6 image",
                ):
                    transform.transform_image(image, True)

    def test_compute_losses_propagates_giou_assertion(self):
        observed_devices = []

        def invalid_giou(pred_boxes, gt_boxes):
            self.assertEqual(pred_boxes.device, gt_boxes.device)
            observed_devices.append(pred_boxes.device.type)
            raise AssertionError("invalid boxes from giou")

        actor = SEATrackActor(
            net=None,
            objective={
                "giou": invalid_giou,
                "l1": lambda *args: torch.tensor(0.0),
                "focal": lambda *args: torch.tensor(0.0),
            },
            loss_weight={"giou": 1.0, "l1": 1.0, "focal": 1.0},
            settings=SimpleNamespace(batchsize=1),
            cfg=SimpleNamespace(
                DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=16)),
                MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(STRIDE=16)),
            ),
        )
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=device.type):
                pred_dict = {
                    "pred_boxes": torch.tensor(
                        [[[0.5, 0.5, 0.25, 0.25]]],
                        device=device,
                    )
                }
                gt_dict = {
                    "search_anno": [
                        torch.tensor(
                            [[0.25, 0.25, 0.25, 0.25]],
                            device=device,
                        )
                    ]
                }

                with self.assertRaisesRegex(AssertionError, "invalid boxes from giou"):
                    actor.compute_losses(pred_dict, gt_dict)

        self.assertEqual(observed_devices, [device.type for device in devices])

    def test_compute_losses_logs_bilift_diagnostics_as_scalars(self):
        actor = SEATrackActor(
            net=None,
            objective={
                "giou": lambda *args: (torch.tensor(0.2), torch.tensor([0.5])),
                "l1": lambda *args: torch.tensor(0.1),
                "focal": lambda *args: torch.tensor(0.3),
            },
            loss_weight={"giou": 1.0, "l1": 1.0, "focal": 1.0},
            settings=SimpleNamespace(batchsize=1),
            cfg=SimpleNamespace(
                DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=16)),
                MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(STRIDE=16)),
            ),
        )
        pred_dict = {
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
            "score_map": torch.zeros(1, 1, 1, 1),
            "bilift_stats": {
                "BiLift/x2r_update_ratio": torch.tensor(0.25, requires_grad=True),
            },
        }
        gt_dict = {
            "search_anno": [torch.tensor([[0.25, 0.25, 0.25, 0.25]])],
        }

        _, status = actor.compute_losses(pred_dict, gt_dict)

        self.assertEqual(status["BiLift/x2r_update_ratio"], 0.25)
        self.assertIsInstance(status["BiLift/x2r_update_ratio"], float)


if __name__ == "__main__":
    unittest.main()
