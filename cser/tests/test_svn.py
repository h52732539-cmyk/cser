from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from cser.svn_model import SubmodularValueNetwork, submodularity_penalty


@unittest.skipIf(torch is None, "torch is not installed")
class TestSVN(unittest.TestCase):
    def test_output_shape(self):
        model = SubmodularValueNetwork(query_dim=16, n_experts=5)
        x = torch.randn(4, 16)
        selected = torch.zeros(4, 5)
        selected[:, 0] = 1.0
        out = model(x, selected)
        self.assertEqual(tuple(out.shape), (4, 5))

    def test_submodularity_penalty(self):
        small = torch.tensor([[0.5, 0.4, 0.3]])
        large_ok = torch.tensor([[0.3, 0.2, 0.1]])
        large_bad = torch.tensor([[0.6, 0.2, 0.4]])
        self.assertAlmostEqual(float(submodularity_penalty(small, large_ok)), 0.0)
        self.assertGreater(float(submodularity_penalty(small, large_bad)), 0.0)


if __name__ == "__main__":
    unittest.main()
