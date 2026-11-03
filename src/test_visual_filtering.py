import numpy as np

from visual_filtering import outlier_ratio, reference_stats


def test_outlier_ratio():
    rng = np.random.default_rng(0)
    grey = rng.normal(180, 0.5, (1000, 3))
    stats = reference_stats(grey, min_std=1.0)
    assert stats["std"] == [1.0, 1.0, 1.0]  # floor applied
    assert outlier_ratio(grey, stats, 3) == 0
    # 20% of frames with the overlay missing (dark parliament footage)
    mixed = grey.copy()
    mixed[:200] = [90, 40, 30]
    assert abs(outlier_ratio(mixed, stats, 3) - 0.2) < 1e-9
    assert outlier_ratio(np.empty((0, 3)), stats, 3) == 1.0


if __name__ == "__main__":
    test_outlier_ratio()
    print("ok")
