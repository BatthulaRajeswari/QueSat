import io

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import regional_visualization as rv
from src.feature_engineering import add_spectral_indices, band_summary


def test_demo_data_shape_and_labels():
    df = dl.generate_demo_data("Krishna", "Water-body Change")
    assert list(df.columns) == dl.REQUIRED
    assert len(df) == 240
    assert set(df[dl.TARGET].unique()) == {0, 1}
    assert df.isna().sum().sum() == 0


def test_demo_data_deterministic_and_region_specific():
    a = dl.generate_demo_data("Krishna", "Flood-related Change")
    b = dl.generate_demo_data("Krishna", "Flood-related Change")
    c = dl.generate_demo_data("Guntur", "Flood-related Change")
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        dl.generate_demo_data("Krishna", "Nope")


def test_scenario_signature_present_in_data():
    df = dl.generate_demo_data("Krishna", "Water-body Change", n=2000)
    m = band_summary(df)
    assert m.loc["Change", "NIR"] < m.loc["No Change", "NIR"]


def _csv(rows, header="Blue,Green,Red,NIR,Change"):
    return io.StringIO(header + "\n" + "\n".join(rows))


def test_valid_csv_accepted_and_case_insensitive():
    rows = [f"0.{i % 9 + 1},0.3,0.2,0.5,{i % 2}" for i in range(40)]
    rep = dl.load_csv(_csv(rows, "blue,GREEN,Red,nir,change"))
    assert rep.ok and rep.stats["n_valid"] == 40
    assert list(rep.df.columns)[:5] == dl.REQUIRED


def test_missing_column_reported():
    rep = dl.load_csv(_csv(["0.1,0.2,0.3,1"], "Blue,Green,Red,Change"))
    assert not rep.ok and "NIR" in rep.errors[0]


def test_nan_rows_dropped_with_warning():
    rows = [f"0.2,0.3,0.2,0.5,{i % 2}" for i in range(40)] + ["abc,0.3,0.2,0.5,1", ",0.3,0.2,0.5,0"]
    rep = dl.load_csv(_csv(rows))
    assert rep.ok and rep.stats["dropped_rows"] == 2 and rep.warnings


def test_too_small_and_single_class_rejected():
    assert not dl.load_csv(_csv(["0.1,0.2,0.3,0.4,1"] * 5)).ok
    rows = ["0.1,0.2,0.3,0.4,1"] * 40
    rep = dl.load_csv(_csv(rows))
    assert not rep.ok and any("No Change" in e for e in rep.errors)


def test_bad_labels_rejected():
    rows = [f"0.1,0.2,0.3,0.4,{i % 3}" for i in range(40)]
    rep = dl.load_csv(_csv(rows))
    assert not rep.ok and "0 (No Change) and 1 (Change)" in rep.errors[0]


def test_empty_and_garbage_files_do_not_crash():
    assert not dl.load_csv(io.StringIO("")).ok
    assert not dl.load_csv(io.BytesIO(b"\xff\xfe\x00\x01garbage")).ok


def test_geo_columns_detected():
    rows = [f"0.2,0.3,0.2,0.5,{i % 2},16.5,80.6" for i in range(40)]
    rep = dl.load_csv(_csv(rows, "Blue,Green,Red,NIR,Change,Latitude,Longitude"))
    assert rep.ok and dl.has_geo(rep.df) and dl.fraction_inside_ap(rep.df) == 1.0


def test_spectral_indices():
    df = pd.DataFrame({"Blue": [0.1], "Green": [0.3], "Red": [0.2], "NIR": [0.6], "Change": [1]})
    out = add_spectral_indices(df)
    assert out["NDVI"].iloc[0] == pytest.approx((0.6 - 0.2) / 0.8, abs=1e-6)
    assert out["NDWI"].iloc[0] == pytest.approx((0.3 - 0.6) / 0.9, abs=1e-6)


@pytest.mark.parametrize("area", dl.AREAS)
@pytest.mark.parametrize("scenario", list(dl.SCENARIOS))
def test_every_scene_has_both_change_and_no_change(area, scenario):
    scene = rv.build_scene(area, scenario)
    frac = scene["truth"].mean()
    assert 0.03 < frac < 0.7
    assert scene["truth"].shape == (rv.SCENE_SIZE, rv.SCENE_SIZE)


def test_shipped_demo_csvs_validate(tmp_path):
    for p in dl.write_demo_csvs(tmp_path):
        assert dl.load_csv(p).ok
