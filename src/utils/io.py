"""Helpers simples pour charger du JSON et préparer des DataFrames lisibles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> Any:
    """Charge un fichier JSON en UTF-8."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_to_df(path: Path, make_view: bool = False) -> pd.DataFrame:
    """Charge un fichier JSON puis le convertit en DataFrame.

    Si `make_view=True`, les colonnes contenant des valeurs complexes
    comme des listes ou des dictionnaires sont converties en texte pour
    faciliter l'inspection dans PyCharm.
    """
    data = load_json(path)
    df = pd.DataFrame(data)

    if make_view:
        return make_dataframe_view(df)

    return df


def format_value_for_display(value: Any) -> Any:
    """Convertit les valeurs complexes en texte pour l'affichage."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, set):
        return str(value)
    return value


def make_dataframe_view(df: pd.DataFrame) -> pd.DataFrame:
    """Crée une copie du DataFrame adaptée à l'affichage."""
    df_view = df.copy()

    for col in df_view.columns:
        has_complex_values = df_view[col].dropna().map(
            lambda x: isinstance(x, (list, dict, set))
        ).any()

        if has_complex_values:
            df_view[col] = df_view[col].apply(format_value_for_display)

    return df_view
