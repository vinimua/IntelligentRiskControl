"""任务三监督标签保护测试。"""

import pandas as pd
import pytest

from apps.modelops_api.services.monitoring.window_loader import (
    WindowContractError,
    validate_window_labels,
)


def test_observed_binary_labels_are_allowed():
    frame = pd.DataFrame({"is_bad": [0, 1, 0]})
    validate_window_labels("W2", frame)


def test_missing_is_bad_is_rejected_without_imputation():
    frame = pd.DataFrame({"is_bad": [0, None, 1]})
    with pytest.raises(WindowContractError, match="never be imputed"):
        validate_window_labels("W2", frame)


def test_non_binary_is_bad_is_rejected():
    frame = pd.DataFrame({"is_bad": [0, 2, 1]})
    with pytest.raises(WindowContractError, match="only 0 or 1"):
        validate_window_labels("W2", frame)
