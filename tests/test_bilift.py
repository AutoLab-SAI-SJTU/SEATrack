import unittest

import torch

from lib.models.layers.bilift import BiLift


class BiLiftTests(unittest.TestCase):
    def test_zero_initialized_bilift_is_exact_identity(self):
        module = BiLift(dim=16, rank=4).eval()
        rgb = torch.randn(2, 5, 16)
        x = torch.randn(2, 5, 16)

        rgb_out, x_out = module(rgb, x)

        self.assertTrue(torch.equal(rgb_out, rgb))
        self.assertTrue(torch.equal(x_out, x))

    def test_invalid_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            BiLift(dim=16, rank=0)

    def test_inverse_reconstructs_both_coupling_orders(self):
        torch.manual_seed(7)
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                module = BiLift(dim=16, rank=4, reverse=reverse).eval()
                torch.nn.init.normal_(module.x_to_rgb.up.weight, std=0.05)
                torch.nn.init.normal_(module.rgb_to_x.up.weight, std=0.05)
                rgb = torch.randn(2, 5, 16)
                x = torch.randn(2, 5, 16)

                rgb_out, x_out = module(rgb, x)
                rgb_reconstructed, x_reconstructed = module.inverse(rgb_out, x_out)

                torch.testing.assert_close(rgb_reconstructed, rgb, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(x_reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_both_cross_updates_receive_finite_gradients(self):
        module = BiLift(dim=16, rank=4)
        torch.nn.init.normal_(module.x_to_rgb.up.weight, std=0.05)
        torch.nn.init.normal_(module.rgb_to_x.up.weight, std=0.05)
        rgb = torch.randn(2, 5, 16, requires_grad=True)
        x = torch.randn(2, 5, 16, requires_grad=True)

        rgb_out, x_out = module(rgb, x)
        (rgb_out.square().mean() + x_out.square().mean()).backward()

        for update in (module.x_to_rgb, module.rgb_to_x):
            self.assertIsNotNone(update.up.weight.grad)
            self.assertTrue(torch.isfinite(update.up.weight.grad).all())

    def test_diagnostics_are_optional_detached_scalars(self):
        rgb = torch.randn(2, 5, 16)
        x = torch.randn(2, 5, 16)
        disabled = BiLift(dim=16, rank=4, diagnostics=False)
        enabled = BiLift(dim=16, rank=4, diagnostics=True)

        disabled(rgb, x)
        enabled(rgb, x)

        self.assertEqual(disabled.last_stats, {})
        self.assertEqual(
            set(enabled.last_stats),
            {
                "BiLift/x2r_update_ratio",
                "BiLift/r2x_update_ratio",
                "BiLift/difference_ratio",
            },
        )
        for value in enabled.last_stats.values():
            self.assertEqual(value.ndim, 0)
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))


if __name__ == "__main__":
    unittest.main()
