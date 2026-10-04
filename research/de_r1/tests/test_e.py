"""CPU-only checks of E's initialization, autograd boundary and checkpoint API."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from research.de_r1.common import LORAT  # Makes the existing native package importable.
from trackit.models.methods.LoRAT.modules.head.mlp import MlpAnchorFreeHead
from research.de_r1.spatial_e import SpatialResidual, SpatialResidualHead, install_spatial


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261004)
        torch.set_num_threads(1)

    def test_zero_residual_preserves_native_score_and_boxes(self):
        head = MlpAnchorFreeHead(16, (5, 3)).eval()
        tokens = torch.randn(2, 15, 16)
        expected = head(tokens)
        wrapped = SpatialResidualHead(head, dim=16, hidden=8, map_size=(5, 3)).eval()
        residual = wrapped.spatial(tokens)
        self.assertEqual(torch.count_nonzero(residual).item(), 0)
        actual = wrapped(tokens)
        for key in ('score_map', 'boxes'):
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        self.assertEqual(actual['score_map'].shape, (2, 3, 5))
        self.assertEqual(actual['boxes'].shape, (2, 3, 5, 4))

    def test_gradient_reaches_spatial_layers_after_first_update(self):
        head = MlpAnchorFreeHead(16, (3, 3))
        wrapped = SpatialResidualHead(head, dim=16, hidden=8, map_size=(3, 3))
        wrapped.train()
        self.assertFalse(head.training)
        self.assertTrue(wrapped.spatial.training)
        frozen = {name: p.detach().clone() for name, p in head.named_parameters()}
        opt = torch.optim.AdamW(wrapped.spatial.parameters(), lr=.01, weight_decay=0.)
        tokens = torch.randn(2, 9, 16)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            out = wrapped(tokens)
            loss = (out['score_map'] - .3).square().mean() + out['boxes'].square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None for p in head.parameters()))
            self.assertGreater(wrapped.spatial.up.weight.grad.abs().sum().item(), 0.)
            if step == 0:
                self.assertEqual(wrapped.spatial.down.weight.grad.abs().sum().item(), 0.)
            else:
                self.assertGreater(wrapped.spatial.down.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(wrapped.spatial.spatial.weight.grad.abs().sum().item(), 0.)
            opt.step()
        self.assertTrue(all(torch.equal(p, frozen[name]) for name, p in head.named_parameters()))

    def test_installation_freezes_B_and_loads_only_branch(self):
        def adapter():
            model = nn.Module()
            model.embed_dim = 16
            model.x_size = (3, 3)
            model.backbone = nn.Linear(16, 16)
            model.head = MlpAnchorFreeHead(16, (3, 3))
            return SimpleNamespace(model=model)

        first = adapter()
        branch = install_spatial(first)
        self.assertIs(branch, first.model.head.spatial)
        self.assertFalse(first.model.training)
        self.assertEqual({id(p) for p in first.model.parameters() if p.requires_grad},
                         {id(p) for p in branch.parameters()})
        with torch.no_grad():
            branch.up.bias.add_(.125)
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'branch.safetensors'
            save_file(branch.state_dict(), str(path))
            restored = install_spatial(adapter(), path)
            self.assertEqual(set(branch.state_dict()), set(restored.state_dict()))
            for name, p in restored.state_dict().items():
                self.assertTrue(torch.equal(p, branch.state_dict()[name]))
        with self.assertRaises(ValueError):
            install_spatial(first)

    def test_branch_budget_dtype_and_spatial_shape(self):
        branch = SpatialResidual()
        self.assertEqual(sum(p.numel() for p in branch.parameters()), 134848)
        tokens = torch.randn(1, 729, 1024)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertEqual(branch(tokens).dtype, tokens.dtype)
        branch = SpatialResidual(16, 8, (3, 3)).to(dtype=torch.bfloat16)
        tiny = torch.randn(1, 9, 16, dtype=torch.bfloat16)
        self.assertEqual(branch(tiny).dtype, tiny.dtype)
        self.assertTrue(torch.equal(tiny, tiny + branch(tiny)))
        with self.assertRaises(ValueError):
            branch(torch.randn(1, 8, 16, dtype=torch.bfloat16))

    def test_neighbor_mixing_preserves_non_square_grid_coordinates(self):
        branch = SpatialResidual(dim=2, hidden=2, map_size=(5, 3))
        with torch.no_grad():
            branch.down.weight.copy_(torch.eye(2))
            branch.down.bias.zero_()
            branch.spatial.weight.zero_()
            branch.spatial.weight[:, 0, 1, 2] = 1.
            branch.spatial.bias.zero_()
            branch.up.weight.copy_(torch.eye(2))
            branch.up.bias.zero_()
        tokens = torch.zeros(1, 15, 2)
        tokens[0, 1 * 5 + 2] = torch.tensor([1., -1.])
        residual = branch(tokens).abs().sum(-1)
        # A kernel that reads its right neighbor moves the response one cell left.
        self.assertEqual(torch.nonzero(residual[0] > 0).flatten().tolist(), [1 * 5 + 1])


if __name__ == '__main__':
    unittest.main()
