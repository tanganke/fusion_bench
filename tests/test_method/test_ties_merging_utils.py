import unittest

import torch

from fusion_bench.method.ties_merging.ties_merging_utils import (
    ties_merging,
    topk_values_mask,
)


class TestTopkValuesMask(unittest.TestCase):
    def test_keep_fraction(self):
        M = torch.randn(3, 100)
        masked, kept = topk_values_mask(M, K=0.2)
        # the k-th largest itself is kept too (>=), as in the original TIES code: 20 or 21 of 100
        self.assertTrue(((kept >= 0.2) & (kept <= 0.21)).all())
        for row, out in zip(M, masked):
            nz = out != 0
            self.assertTrue(torch.equal(out[nz], row[nz]))
            self.assertGreaterEqual(row.abs()[nz].min(), row.abs()[~nz].max())

    def test_keep_all(self):
        """K = 1 (a fraction) and K = 100 (a percentage) keep every element."""
        M = torch.randn(3, 10)
        for K in (1.0, 100):
            masked, kept = topk_values_mask(M, K=K)
            self.assertTrue(torch.equal(masked, M))
            self.assertTrue(torch.equal(kept, torch.ones(3)))

    def test_ties_merging_without_trimming(self):
        """With nothing trimmed, TIES keeps the entries that agree with the sign of the sum."""
        tv = torch.randn(3, 50)
        agree = tv.sign() == tv.sum(0).sign()
        merged = ties_merging(tv, reset_thresh=100, merge_func="sum")
        torch.testing.assert_close(merged, (tv * agree).sum(0))


if __name__ == "__main__":
    unittest.main()
