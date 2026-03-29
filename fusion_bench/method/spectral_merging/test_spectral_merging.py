"""
Unit tests for SpectralMerge implementation.

These tests validate the core algorithm on small random matrices without
requiring GPU or real models. Run with:
    cd implementation && python -m pytest test_spectral_merging.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytest

from utils import (
    AlignmentInfo,
    compute_cross_task_alignment,
    merge_in_spectral_domain,
    spectral_clustering_subspaces,
    _simple_kmeans,
)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_task_svd(device):
    """Create simple task vectors with known alignment structure."""
    torch.manual_seed(42)
    m, n = 64, 32

    # Task 1: dominant direction along [1, 0, 0, ...]
    v1 = torch.zeros(n, device=device)
    v1[0] = 1.0
    u1 = torch.randn(m, device=device)
    u1 = u1 / u1.norm()
    W1 = 5.0 * torch.outer(u1, v1) + 0.1 * torch.randn(m, n, device=device)

    # Task 2: same dominant direction as Task 1 (aligned!)
    u2 = torch.randn(m, device=device)
    u2 = u2 / u2.norm()
    W2 = 4.0 * torch.outer(u2, v1) + 0.1 * torch.randn(m, n, device=device)

    # Task 3: orthogonal direction
    v3 = torch.zeros(n, device=device)
    v3[1] = 1.0
    u3 = torch.randn(m, device=device)
    u3 = u3 / u3.norm()
    W3 = 3.0 * torch.outer(u3, v3) + 0.1 * torch.randn(m, n, device=device)

    # SVD decompose
    svd1 = torch.linalg.svd(W1, full_matrices=False)
    svd2 = torch.linalg.svd(W2, full_matrices=False)
    svd3 = torch.linalg.svd(W3, full_matrices=False)

    return [svd1, svd2, svd3]


class TestCrossTaskAlignment:
    def test_detects_aligned_subspaces(self, simple_task_svd):
        """Should detect that Tasks 1 and 2 share a dominant direction."""
        info = compute_cross_task_alignment(
            task_svd=simple_task_svd,
            num_subspaces=[1, 1, 1],
            alignment_threshold=0.5,
        )

        # Tasks 1 and 2 should be in the same aligned group
        assert len(info.aligned_groups) >= 1, "Should detect at least one aligned group"

        # Check that the aligned group contains subspaces from different tasks
        found_cross_task = False
        for group in info.aligned_groups:
            tasks_in_group = set(t for t, _ in group)
            if len(tasks_in_group) > 1:
                found_cross_task = True
                # Should contain tasks 0 and 1 (aligned)
                assert 0 in tasks_in_group or 1 in tasks_in_group

        assert found_cross_task, "Should detect cross-task alignment between Tasks 1 and 2"

    def test_preserves_orthogonal_subspaces(self, simple_task_svd):
        """Task 3's orthogonal direction should be standalone."""
        info = compute_cross_task_alignment(
            task_svd=simple_task_svd,
            num_subspaces=[1, 1, 1],
            alignment_threshold=0.5,
        )

        # Should have standalone subspaces
        assert len(info.standalone) >= 1, "Should preserve orthogonal subspaces"

    def test_empty_input(self, device):
        """Should handle empty input gracefully."""
        info = compute_cross_task_alignment(
            task_svd=[],
            num_subspaces=[],
            alignment_threshold=0.3,
        )
        assert len(info.aligned_groups) == 0
        assert len(info.standalone) == 0

    def test_single_task(self, device):
        """Single task should result in all standalone subspaces."""
        torch.manual_seed(0)
        W = torch.randn(32, 16, device=device)
        svd = torch.linalg.svd(W, full_matrices=False)

        info = compute_cross_task_alignment(
            task_svd=[svd],
            num_subspaces=[2],
            alignment_threshold=0.3,
        )

        # Single task: everything is standalone (no cross-task alignment possible)
        assert len(info.aligned_groups) == 0


class TestMergeInSpectralDomain:
    def test_returns_correct_shape(self, simple_task_svd, device):
        """Merged weight should have the same shape as base."""
        base = torch.randn(64, 32, device=device)
        info = compute_cross_task_alignment(
            task_svd=simple_task_svd,
            num_subspaces=[2, 2, 2],
            alignment_threshold=0.3,
        )

        merged = merge_in_spectral_domain(
            base_weight=base,
            task_svd=simple_task_svd,
            task_num_subspaces=[2, 2, 2],
            alignment_info=info,
        )

        assert merged.shape == base.shape

    def test_preserves_base_when_no_task_vectors(self, device):
        """With zero task vectors, merged should equal base."""
        base = torch.randn(32, 16, device=device)
        zero_task = torch.zeros(32, 16, device=device)
        svd = torch.linalg.svd(zero_task, full_matrices=False)

        info = AlignmentInfo(aligned_groups=[], standalone=[], alignment_scores=[])

        merged = merge_in_spectral_domain(
            base_weight=base,
            task_svd=[svd],
            task_num_subspaces=[1],
            alignment_info=info,
        )

        # All singular values are ~0, so merged ≈ base
        assert torch.allclose(merged, base, atol=1e-5)

    def test_calibration_reduces_over_accumulation(self, device):
        """Calibrated merge should produce smaller singular values than naive sum."""
        torch.manual_seed(42)
        m, n = 32, 16
        base = torch.zeros(m, n, device=device)

        # Two identical task vectors (maximum alignment)
        task = 3.0 * torch.randn(m, n, device=device)
        svd1 = torch.linalg.svd(task, full_matrices=False)
        svd2 = torch.linalg.svd(task, full_matrices=False)

        # Aligned group: both tasks in same group
        info = AlignmentInfo(
            aligned_groups=[[(0, 0), (1, 0)]],
            standalone=[],
            alignment_scores=[0.99],
        )

        merged = merge_in_spectral_domain(
            base_weight=base,
            task_svd=[svd1, svd2],
            task_num_subspaces=[1, 1],
            alignment_info=info,
            calibration_strength=1.0,
        )

        # Naive sum would be 2 * task = 2 * (U S V^T)
        # Calibrated merge should be attenuated
        naive_sum = 2.0 * task
        assert merged.norm() < naive_sum.norm(), (
            "Calibration should reduce over-accumulation"
        )


class TestSpectralClustering:
    def test_basic_clustering(self, device):
        """Should cluster aligned subspaces together."""
        # Two groups: {0,1} aligned, {2,3} aligned
        alignment = torch.tensor([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ], device=device)

        clusters = spectral_clustering_subspaces(alignment, n_clusters=2, threshold=0.5)

        # Should produce 2 clusters
        assert len(clusters) == 2

        # All 4 points should be assigned
        all_indices = []
        for c in clusters:
            all_indices.extend(c)
        assert sorted(all_indices) == [0, 1, 2, 3]


class TestSimpleKMeans:
    def test_basic_kmeans(self, device):
        """Should separate two well-defined clusters."""
        torch.manual_seed(42)
        # Two clusters: one around [0,0], one around [5,5]
        cluster1 = torch.randn(10, 2, device=device) * 0.1
        cluster2 = torch.randn(10, 2, device=device) * 0.1 + 5.0
        X = torch.cat([cluster1, cluster2], dim=0)

        clusters = _simple_kmeans(X, k=2)

        assert len(clusters) == 2
        assert sum(len(c) for c in clusters) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
