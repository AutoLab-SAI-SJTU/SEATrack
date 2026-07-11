import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from lib.config.seatrack.config import cfg, update_config_from_file
from lib.models.layers.attn_blocks import CEBlock_AP
from lib.models.seatrack.seatrack import build_seatrack
from lib.models.seatrack.vit import _init_vit_weights
from lib.models.seatrack.vit_ci import VisionTransformerCE
from lib.train.base_functions import get_optimizer_scheduler


class BiLiftBlockIntegrationTests(unittest.TestCase):
    @staticmethod
    def _make_clean_block(bilift_enabled):
        return CEBlock_AP(
            dim=32,
            num_heads=4,
            layer=1,
            lora_layers=[1],
            moe_layers=[1],
            amglora_rank=4,
            hmoe_rank=2,
            amg_enabled=False,
            hmoe_enabled=False,
            bilift_enabled=bilift_enabled,
            bilift_rank=4,
        )

    def test_clean_lift_block_omits_amg_and_hmoe_parameters(self):
        block = CEBlock_AP(
            dim=32,
            num_heads=4,
            layer=1,
            lora_layers=[1],
            moe_layers=[1],
            amglora_rank=4,
            hmoe_rank=2,
            amg_enabled=False,
            hmoe_enabled=False,
            bilift_enabled=True,
            bilift_rank=4,
        )

        self.assertFalse(hasattr(block, "attn_moe"))
        self.assertFalse(hasattr(block, "ffn_moe"))
        self.assertFalse(hasattr(block, "r2dte_scaling"))
        self.assertFalse(hasattr(block, "dte2r_scaling"))
        self.assertIsNotNone(block.bilift)
        self.assertEqual(block.attn.qkv.r, 4)

    def test_default_flags_preserve_legacy_modules(self):
        block = CEBlock_AP(
            dim=32,
            num_heads=4,
            layer=1,
            lora_layers=[1],
            moe_layers=[1],
            amglora_rank=4,
            hmoe_rank=2,
        )

        self.assertTrue(hasattr(block, "attn_moe"))
        self.assertTrue(hasattr(block, "ffn_moe"))
        self.assertTrue(hasattr(block, "r2dte_scaling"))
        self.assertTrue(hasattr(block, "dte2r_scaling"))

    def test_zero_initialized_bilift_preserves_block_outputs_exactly(self):
        torch.manual_seed(11)
        baseline = self._make_clean_block(bilift_enabled=False)
        lifted = self._make_clean_block(bilift_enabled=True)
        incompatible = lifted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(all(key.startswith("bilift.") for key in incompatible.missing_keys))
        baseline.eval()
        lifted.eval()

        rgb = torch.randn(2, 5, 32)
        x = torch.randn(2, 5, 32)
        template_index = torch.arange(2).repeat(2, 1)
        search_index = torch.arange(3).repeat(2, 1)

        baseline_out = baseline(
            [rgb.clone(), x.clone()],
            template_index.clone(),
            [search_index.clone(), search_index.clone()],
        )
        lifted_out = lifted(
            [rgb.clone(), x.clone()],
            template_index.clone(),
            [search_index.clone(), search_index.clone()],
        )

        for baseline_tokens, lifted_tokens in zip(baseline_out[0], lifted_out[0]):
            self.assertTrue(torch.equal(baseline_tokens, lifted_tokens))
        self.assertTrue(torch.equal(baseline_out[1], lifted_out[1]))
        for baseline_indices, lifted_indices in zip(baseline_out[2], lifted_out[2]):
            self.assertTrue(torch.equal(baseline_indices, lifted_indices))
        self.assertEqual(baseline_out[3], lifted_out[3])


class BiLiftModelIntegrationTests(unittest.TestCase):
    @staticmethod
    def _make_tiny_backbone(diagnostics=False):
        return VisionTransformerCE(
            img_size=32,
            patch_size=16,
            embed_dim=32,
            depth=4,
            num_heads=4,
            mlp_ratio=2,
            search_size=(32, 32),
            template_size=(16, 16),
            new_patch_size=16,
            ce_loc=[],
            ce_keep_ratio=[],
            amglora_rank=4,
            hmoe_rank=2,
            amg_enabled=False,
            hmoe_enabled=False,
            bilift_enabled=True,
            bilift_layers=[1, 3],
            bilift_rank=4,
            bilift_diagnostics=diagnostics,
        )

    def test_config_defaults_preserve_legacy_model(self):
        self.assertTrue(cfg.MODEL.AMG_ENABLED)
        self.assertTrue(cfg.MODEL.HMOE_ENABLED)
        self.assertFalse(cfg.MODEL.DETERMINISTIC_LORA_INIT)
        self.assertFalse(cfg.MODEL.BILIFT.ENABLED)
        self.assertEqual(cfg.MODEL.BILIFT.LAYERS, [5, 9])
        self.assertEqual(cfg.MODEL.BILIFT.RANK, 8)
        self.assertEqual(cfg.MODEL.BILIFT.DROPOUT, 0.0)
        self.assertFalse(cfg.MODEL.BILIFT.DIAGNOSTICS)

    def test_named_lora_initialization_is_architecture_independent(self):
        from lib.models.seatrack.seatrack import _reset_lora_parameters_by_name

        def make_stack(hmoe_enabled, bilift_enabled):
            return nn.Sequential(
                *[
                    CEBlock_AP(
                        dim=32,
                        num_heads=4,
                        layer=layer,
                        lora_layers=[1, 3],
                        moe_layers=[1, 3],
                        amglora_rank=4,
                        hmoe_rank=2,
                        amg_enabled=False,
                        hmoe_enabled=hmoe_enabled,
                        bilift_enabled=bilift_enabled,
                        bilift_rank=4,
                    )
                    for layer in (1, 3)
                ]
            )

        torch.manual_seed(0)
        legacy = make_stack(hmoe_enabled=True, bilift_enabled=False)
        torch.manual_seed(0)
        lifted = make_stack(hmoe_enabled=False, bilift_enabled=True)
        legacy_before = {
            name: parameter.detach().clone()
            for name, parameter in legacy.named_parameters()
            if ".attn.qkv.lora_" in name
        }
        lifted_before = {
            name: parameter.detach().clone()
            for name, parameter in lifted.named_parameters()
            if ".attn.qkv.lora_" in name
        }
        self.assertTrue(
            any(not torch.equal(legacy_before[name], lifted_before[name]) for name in legacy_before)
        )

        _reset_lora_parameters_by_name(legacy, seed=42)
        _reset_lora_parameters_by_name(lifted, seed=42)
        legacy_after = dict(legacy.named_parameters())
        lifted_after = dict(lifted.named_parameters())
        lora_names = [name for name in legacy_after if ".attn.qkv.lora_" in name]
        self.assertTrue(lora_names)
        for name in lora_names:
            self.assertTrue(torch.equal(legacy_after[name], lifted_after[name]), name)

        seed_42 = legacy_after[lora_names[0]].detach().clone()
        _reset_lora_parameters_by_name(legacy, seed=43)
        self.assertFalse(torch.equal(seed_42, dict(legacy.named_parameters())[lora_names[0]]))

    def test_vit_initializer_supports_parameter_free_layer_norm(self):
        norm = nn.LayerNorm(16, elementwise_affine=False)

        _init_vit_weights(norm)

        self.assertIsNone(norm.weight)
        self.assertIsNone(norm.bias)

    def test_lifttrack_rejects_legacy_cross_modal_modules(self):
        conflicts = (
            ("AMG", {"amg": True}),
            ("HMoE", {"hmoe": True}),
            ("GRA", {"gra": True}),
            ("GRA diagnostics", {"diagnostics": True}),
        )

        for label, enabled in conflicts:
            with self.subTest(conflict=label):
                test_cfg = deepcopy(cfg)
                test_cfg.MODEL.PRETRAIN_FILE = ""
                test_cfg.MODEL.BILIFT.ENABLED = True
                test_cfg.MODEL.AMG_ENABLED = enabled.get("amg", False)
                test_cfg.MODEL.HMOE_ENABLED = enabled.get("hmoe", False)
                test_cfg.MODEL.GRA.ENABLED = enabled.get("gra", False)
                test_cfg.MODEL.GRA.DIAGNOSTICS = enabled.get("diagnostics", False)

                with self.assertRaisesRegex(ValueError, label):
                    build_seatrack(test_cfg, training=False)

    def test_backbone_selects_layers_and_alternates_coupling_order(self):
        backbone = self._make_tiny_backbone()

        self.assertFalse(hasattr(backbone.blocks[0], "bilift"))
        self.assertTrue(hasattr(backbone.blocks[1], "bilift"))
        self.assertFalse(backbone.blocks[1].bilift.reverse)
        self.assertFalse(hasattr(backbone.blocks[2], "bilift"))
        self.assertTrue(hasattr(backbone.blocks[3], "bilift"))
        self.assertTrue(backbone.blocks[3].bilift.reverse)
        for block in backbone.blocks:
            self.assertFalse(hasattr(block, "attn_moe"))
            self.assertFalse(hasattr(block, "ffn_moe"))
            self.assertFalse(hasattr(block, "r2dte_scaling"))
            self.assertFalse(hasattr(block, "dte2r_scaling"))

    def test_builder_propagates_lifttrack_configuration(self):
        class DummyBackbone(nn.Module):
            embed_dim = 32

        class DummyHead(nn.Module):
            feat_sz = 1

        test_cfg = deepcopy(cfg)
        test_cfg.MODEL.PRETRAIN_FILE = ""
        test_cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224_ce"
        test_cfg.MODEL.AMG_ENABLED = False
        test_cfg.MODEL.HMOE_ENABLED = False
        test_cfg.MODEL.BILIFT.ENABLED = True

        with patch(
            "lib.models.seatrack.seatrack.vit_base_patch16_224_ce",
            return_value=DummyBackbone(),
        ) as backbone_factory, patch(
            "lib.models.seatrack.seatrack.build_box_head",
            return_value=DummyHead(),
        ):
            build_seatrack(test_cfg, training=False)

        kwargs = backbone_factory.call_args.kwargs
        self.assertFalse(kwargs["amg_enabled"])
        self.assertFalse(kwargs["hmoe_enabled"])
        self.assertTrue(kwargs["bilift_enabled"])
        self.assertEqual(kwargs["bilift_layers"], [5, 9])
        self.assertEqual(kwargs["bilift_rank"], 8)
        self.assertEqual(kwargs["bilift_dropout"], 0.0)
        self.assertFalse(kwargs["bilift_diagnostics"])

    def test_builder_uses_experiment_seed_for_named_lora_initialization(self):
        class DummyBackbone(nn.Module):
            embed_dim = 32

        class DummyHead(nn.Module):
            feat_sz = 1

        test_cfg = deepcopy(cfg)
        test_cfg.MODEL.PRETRAIN_FILE = ""
        test_cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224_ce"
        test_cfg.MODEL.DETERMINISTIC_LORA_INIT = True

        with patch(
            "lib.models.seatrack.seatrack.vit_base_patch16_224_ce",
            return_value=DummyBackbone(),
        ), patch(
            "lib.models.seatrack.seatrack.build_box_head",
            return_value=DummyHead(),
        ), patch(
            "lib.models.seatrack.seatrack._reset_lora_parameters_by_name",
        ) as reset_lora:
            model = build_seatrack(
                test_cfg,
                training=False,
                settings=SimpleNamespace(seed=23),
            )

        reset_lora.assert_called_once_with(model, seed=23)

    def test_backbone_aggregates_detached_bilift_diagnostics(self):
        backbone = self._make_tiny_backbone(diagnostics=True).eval()
        template = torch.randn(2, 6, 16, 16)
        search = torch.randn(2, 6, 32, 32)

        with torch.no_grad():
            _, aux_dict = backbone(template, search)

        self.assertEqual(
            set(aux_dict["bilift_stats"]),
            {
                "BiLift/x2r_update_ratio",
                "BiLift/r2x_update_ratio",
                "BiLift/difference_ratio",
            },
        )
        for value in aux_dict["bilift_stats"].values():
            self.assertEqual(value.ndim, 0)
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))

        without_diagnostics = self._make_tiny_backbone(diagnostics=False).eval()
        with torch.no_grad():
            _, aux_without_diagnostics = without_diagnostics(template, search)
        self.assertNotIn("bilift_stats", aux_without_diagnostics)

    def test_peft_selects_only_lora_and_bilift_parameters(self):
        backbone = VisionTransformerCE(
            img_size=32,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=1,
            search_size=(32, 32),
            template_size=(16, 16),
            new_patch_size=16,
            ce_loc=[],
            ce_keep_ratio=[],
            amglora_rank=8,
            hmoe_rank=4,
            amg_enabled=False,
            hmoe_enabled=False,
            bilift_enabled=True,
            bilift_layers=[5, 9],
            bilift_rank=8,
        )
        test_cfg = deepcopy(cfg)
        test_cfg.TRAIN.PEFT = True

        optimizer, _ = get_optimizer_scheduler(backbone, test_cfg)

        trainable = {
            name: parameter
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        }
        bilift_count = sum(
            parameter.numel()
            for name, parameter in trainable.items()
            if "bilift" in name
        )
        self.assertEqual(bilift_count, 49_152)
        self.assertEqual(sum(parameter.numel() for parameter in trainable.values()), 196_608)
        self.assertTrue(any("lora" in name for name in trainable))
        self.assertTrue(any("bilift" in name for name in trainable))
        for forbidden in ("moe", "rgae", "r2dte_scaling", "dte2r_scaling"):
            self.assertFalse(any(forbidden in name for name in trainable))

        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in trainable.values()})


class LiftTrackExperimentConfigTests(unittest.TestCase):
    config_dir = Path(__file__).resolve().parents[1] / "experiments" / "seatrack"

    def _load(self, name):
        loaded = deepcopy(cfg)
        update_config_from_file(str(self.config_dir / f"{name}.yaml"), loaded)
        return loaded

    def test_method_flags_are_explicit_and_gra_is_disabled(self):
        expected_flags = {
            "rgbt_lora_only": (False, False, False),
            "rgbt_seatrack_pilot": (True, True, False),
            "rgbt_lora_only_pilot": (False, False, False),
            "rgbt_lifttrack_short": (False, False, True),
            "rgbt_lifttrack_pilot": (False, False, True),
            "rgbt_lifttrack": (False, False, True),
        }

        for name, expected in expected_flags.items():
            with self.subTest(config=name):
                loaded = self._load(name)
                actual = (
                    loaded.MODEL.AMG_ENABLED,
                    loaded.MODEL.HMOE_ENABLED,
                    loaded.MODEL.BILIFT.ENABLED,
                )
                self.assertEqual(actual, expected)
                self.assertFalse(loaded.MODEL.GRA.ENABLED)
                self.assertFalse(loaded.MODEL.GRA.DIAGNOSTICS)
                self.assertTrue(loaded.MODEL.DETERMINISTIC_LORA_INIT)
                self.assertEqual(loaded.MODEL.BILIFT.LAYERS, [5, 9])
                self.assertEqual(loaded.MODEL.BILIFT.RANK, 8)
                self.assertEqual(loaded.MODEL.BILIFT.DROPOUT, 0.0)

    def test_short_pilot_and_full_protocols_are_matched(self):
        short = self._load("rgbt_lifttrack_short")
        self.assertEqual(short.DATA.TRAIN.DATASETS_NAME, ["LasHeR_smoke"])
        self.assertEqual(short.DATA.TRAIN.SAMPLE_PER_EPOCH, 1)
        self.assertEqual(short.DATA.VAL.DATASETS_NAME, [None])
        self.assertEqual(short.TRAIN.BATCH_SIZE, 1)
        self.assertEqual(short.TRAIN.EPOCH, 1)
        self.assertEqual(short.TRAIN.VAL_EPOCH_INTERVAL, 1)
        self.assertEqual(short.TRAIN.SAVE_EPOCH_INTERVAL, 1)

        pilot_names = (
            "rgbt_seatrack_pilot",
            "rgbt_lora_only_pilot",
            "rgbt_lifttrack_pilot",
        )
        pilots = [self._load(name) for name in pilot_names]
        for loaded in pilots:
            self.assertEqual(loaded.DATA.TRAIN.DATASETS_NAME, ["LasHeR_train"])
            self.assertEqual(loaded.DATA.TRAIN.SAMPLE_PER_EPOCH, 60_000)
            self.assertEqual(loaded.DATA.VAL.DATASETS_NAME, ["LasHeR_val"])
            self.assertEqual(loaded.DATA.VAL.SAMPLE_PER_EPOCH, 60_000)
            self.assertEqual(loaded.TRAIN.BATCH_SIZE, 32)
            self.assertEqual(loaded.TRAIN.EPOCH, 5)
            self.assertEqual(loaded.TRAIN.VAL_EPOCH_INTERVAL, 1)
            self.assertEqual(loaded.TRAIN.SAVE_EPOCH_INTERVAL, 1)
            self.assertEqual(loaded.TRAIN.SAVE_LAST_N_EPOCH, 5)
        pilot_protocol = (
            "MODEL.PRETRAIN_FILE",
            "DATA.SEARCH.SIZE",
            "DATA.TEMPLATE.SIZE",
            "TRAIN.LR",
            "TRAIN.WEIGHT_DECAY",
            "TRAIN.LR_DROP_EPOCH",
            "TRAIN.NUM_WORKER",
        )
        for field in pilot_protocol:
            parts = field.split(".")
            values = []
            for loaded in pilots:
                value = loaded
                for part in parts:
                    value = getattr(value, part)
                values.append(value)
            self.assertEqual(values, [values[0]] * len(values), field)

        for name in ("rgbt_lora_only", "rgbt_lifttrack"):
            loaded = self._load(name)
            self.assertEqual(loaded.DATA.TRAIN.SAMPLE_PER_EPOCH, 60_000)
            self.assertEqual(loaded.DATA.VAL.SAMPLE_PER_EPOCH, 60_000)
            self.assertEqual(loaded.TRAIN.BATCH_SIZE, 32)
            self.assertEqual(loaded.TRAIN.EPOCH, 60)
            self.assertEqual(loaded.TRAIN.VAL_EPOCH_INTERVAL, 5)
            self.assertEqual(loaded.TRAIN.SAVE_EPOCH_INTERVAL, 5)
            self.assertEqual(loaded.TRAIN.SAVE_LAST_N_EPOCH, 5)


if __name__ == "__main__":
    unittest.main()
