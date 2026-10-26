import pytest

from prime_rl.trainer.utils import loss_log_scale


@pytest.mark.parametrize("num_micro_batches,dp_size", [(1, 1), (2, 1), (1, 2), (3, 4)])
def test_loss_log_scale_recovers_global_objective(num_micro_batches, dp_size):
    """The mean TensorStats reports is the objective divided by both dimensions."""
    objective = 6.0
    per_appended_value = objective / (num_micro_batches * dp_size)
    appended = [per_appended_value] * (num_micro_batches * dp_size)

    reported = sum(appended) / len(appended)

    assert reported * loss_log_scale(num_micro_batches, dp_size) == pytest.approx(objective)


def test_loss_log_scale_does_not_double_count_cp_copies():
    """CP ranks append the same per-dp-group loss, so scaling must not include cp."""
    dp_size, cp_size = 2, 2
    per_dp_group = [3.0, 5.0]
    appended = [loss for loss in per_dp_group for _ in range(cp_size)]

    reported = sum(appended) / len(appended)

    assert reported * loss_log_scale(1, dp_size) == pytest.approx(sum(per_dp_group))
