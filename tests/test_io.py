import json
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_json, load_json_to_df


def test_load_json_reads_file(tmp_path):
    path = tmp_path / "sample.json"
    payload = {"name": "Echo", "count": 2}
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_json(path) == payload


def test_load_json_to_df_returns_dataframe(tmp_path):
    path = tmp_path / "events.json"
    payload = [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = load_json_to_df(path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_json_to_df_make_view_converts_complex_values(tmp_path):
    path = tmp_path / "events.json"
    payload = [{"id": 1, "meta": {"city": "Lanton"}, "tags": ["Nature", "Ciel"]}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = load_json_to_df(path, make_view=True)

    assert isinstance(df.loc[0, "meta"], str)
    assert isinstance(df.loc[0, "tags"], str)
