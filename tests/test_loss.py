"""Unit tests for dojo/loss.py — no GPU required."""
import pytest
import torch
from dojo.loss import normalize_per_group, grpo_loss


class TestNormalizePerGroup:
    def test_basic_normalization(self):
        # Two groups of 4, manually checkable
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0,   # group 1: mean=2.5
                                 10.0, 10.0, 10.0, 14.0])  # group 2: mean=11
        adv = normalize_per_group(rewards, group_size=4)
        assert adv.shape == (8,)

        # Group 1 mean should be 0
        assert abs(adv[:4].mean().item()) < 1e-5

        # Group 2 mean should be 0
        assert abs(adv[4:].mean().item()) < 1e-5

    def test_degenerate_all_same_reward(self):
        # All episodes in a group have the same reward → std=0 → advantages=0
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
        adv = normalize_per_group(rewards, group_size=4)
        assert torch.all(adv == 0.0)

    def test_degenerate_two_groups_all_same(self):
        rewards = torch.tensor([3.0, 3.0, 7.0, 7.0])
        adv = normalize_per_group(rewards, group_size=2)
        assert torch.all(adv == 0.0)

    def test_invalid_group_size(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            normalize_per_group(rewards, group_size=2)

    def test_output_shape_multi_group(self):
        n, g = 16, 4
        rewards = torch.randn(n)
        adv = normalize_per_group(rewards, g)
        assert adv.shape == (n,)


class TestGRPOLoss:
    def _make_inputs(self, n=4, l=8):
        """Helper to create small synthetic inputs."""
        log_probs = torch.randn(n, l) * 0.1         # small values around 0
        ref_log_probs = torch.randn(n, l) * 0.1
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.ones(n, l)
        mask[:, :2] = 0   # first 2 positions are prompt
        return log_probs, ref_log_probs, advantages, mask

    def test_returns_scalar(self):
        lp, rlp, adv, mask = self._make_inputs()
        loss = grpo_loss(lp, rlp, adv, mask)
        assert loss.shape == ()

    def test_finite_loss(self):
        lp, rlp, adv, mask = self._make_inputs()
        loss = grpo_loss(lp, rlp, adv, mask)
        assert torch.isfinite(loss)

    def test_zero_mask_gives_zero_loss(self):
        lp, rlp, adv, mask = self._make_inputs()
        zero_mask = torch.zeros_like(mask)
        loss = grpo_loss(lp, rlp, adv, zero_mask)
        # With all-zero mask, numerator=0, denominator=1e-8, so loss ≈ 0
        assert abs(loss.item()) < 1e-5

    def test_beta_penalty(self):
        lp, rlp, adv, mask = self._make_inputs()
        loss_no_kl = grpo_loss(lp, rlp, adv, mask, beta=0.0)
        loss_with_kl = grpo_loss(lp, rlp, adv, mask, beta=0.01)
        # They should differ when log_probs != ref_log_probs
        assert not torch.isclose(loss_no_kl, loss_with_kl, atol=1e-6)

    def test_zero_advantages_zero_policy_gradient(self):
        """With all-zero advantages, policy gradient term = 0."""
        n, l = 4, 8
        log_probs = torch.zeros(n, l)
        ref_log_probs = torch.zeros(n, l)
        advantages = torch.zeros(n)
        mask = torch.ones(n, l)
        loss = grpo_loss(log_probs, ref_log_probs, advantages, mask, beta=0.0)
        assert abs(loss.item()) < 1e-6

    def test_identical_policy_ratio_is_one(self):
        """When log_probs == ref_log_probs, ratio=1, loss = -min(adv, adv) = -adv (clipped)."""
        n, l = 2, 4
        lp = torch.zeros(n, l)
        rlp = torch.zeros(n, l)
        adv = torch.tensor([1.0, -1.0])
        mask = torch.ones(n, l)
        loss = grpo_loss(lp, rlp, adv, mask, epsilon=0.2, beta=0.0)
        # ratio=1 everywhere, -min(1*adv, 1*adv) = -adv
        # mean over tokens: for adv[0]=1 → pg=-1; adv[1]=-1 → pg=1
        # total: (-1*4 + 1*4) / 8 = 0
        assert abs(loss.item()) < 1e-5
