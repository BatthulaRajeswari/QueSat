import numpy as np
import pytest

from src.data_loader import FEATURES, generate_demo_data
from src.feature_engineering import add_spectral_indices, feature_columns
from src.preprocessing import preprocess


@pytest.fixture(scope="module")
def df():
    return generate_demo_data("Krishna", "Water-body Change")


def test_default_four_features_no_pca(df):
    p = preprocess(df, FEATURES)
    assert p.n_components == 4 and not p.used_pca and p.pca is None
    assert p.X_train.shape[1] == 4


def test_angles_in_range_and_split_sizes(df):
    p = preprocess(df, FEATURES, test_size=0.25)
    for X in (p.X_train, p.X_test):
        assert X.min() >= 0 and X.max() <= np.pi + 1e-9
    assert len(p.y_train) + len(p.y_test) == len(df)
    assert abs(len(p.y_test) / len(df) - 0.25) < 0.02
    assert set(p.y_test) == {0, 1}


def test_scaler_fit_on_train_only(df):
    p = preprocess(df, FEATURES)
    assert np.allclose(p.X_train_scaled.mean(axis=0), 0, atol=1e-9)


def test_pca_used_only_when_required(df):
    p = preprocess(df, FEATURES, n_components=2)
    assert p.used_pca and p.X_train.shape[1] == 2
    assert len(p.explained_variance) == 2


def test_indices_give_six_features_and_pca_to_four(df):
    d2 = add_spectral_indices(df)
    cols = feature_columns(True)
    p = preprocess(d2, cols, n_components=4)
    assert p.n_input_features == 6 and p.used_pca and p.n_components == 4


def test_component_request_clamped(df):
    p = preprocess(df, FEATURES, n_components=9)
    assert p.n_components == 4 and any("adjusted" in n for n in p.notes)


def test_transform_matches_stored_test_features(df):
    p = preprocess(df, FEATURES, n_components=3)
    raw = df.iloc[p.idx_test][FEATURES].to_numpy()
    assert np.allclose(p.transform(raw), p.X_test)


def test_nan_rejected(df):
    bad = df.copy()
    bad.loc[0, "Blue"] = np.nan
    with pytest.raises(ValueError):
        preprocess(bad, FEATURES)
