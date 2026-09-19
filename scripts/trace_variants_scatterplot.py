# PLOTINPUT_alignment_traces_BPI_Challenge_2012_noise0.2_max_number_traces100_trace_length177.csv
# PLOTINPUT_alignment_traces_Sepsis_Cases_-_Event_Log_noise0.2_max_number_traces1000_trace_length191.csv
#
# a_star_alignments_Sepsis_Cases_-_Event_Log.xes.csv
# a_star_alignments_BPI_Challenge_2012.xes.csv

#!/usr/bin/env python3
"""
Plotly Scatter: Tracelänge (x) vs. Berechnungszeit (y)
für zwei CSV-Dateien, Ausgabe als HTML.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px

from scripts.evaluate_ibf_alignment_pipeline import log_name

# -----------------------------
# Platzhalter-Konfiguration
# -----------------------------
log_name = "Sepsis Cases"
#log_name = "BPI 2012"

CSV_1_PATH = "../output/a_star_alignments_Sepsis_Cases_-_Event_Log.xes.csv"
CSV_2_PATH = "../output/PLOTINPUT_alignment_traces_Sepsis_Cases_-_Event_Log_noise0.2_max_number_traces1000_trace_length191.csv"

#CSV_1_PATH = "../output/a_star_alignments_BPI_Challenge_2012.xes.csv"
#CSV_2_PATH = "../output/PLOTINPUT_alignment_traces_BPI_Challenge_2012_noise0.2_max_number_traces100_trace_length177.csv"

CSV_1_X_COL = "variant"
CSV_1_Y_COL = "alignment_time"
CSV_1_GROUP_NAME = "A*"

CSV_2_X_COL = "given trace"
CSV_2_Y_COL = "calculation time [ms]"
CSV_2_GROUP_NAME = "IBF based"

GROUP_COLORS = {
    "A*": "#86beda",
    "IBF based": "#6756be",

}

OUTPUT_HTML_PATH = "../output/plots/" + log_name + "_time_by_length.html"

TRACE_SEPARATOR = " - "   # Trennzeichen in der Trace


def _read_and_prepare(
    csv_path: str,
    trace_col: str,
    time_col: str,
    group_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")

    missing = [c for c in (trace_col, time_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Fehlende Spalten in '{csv_path}': {missing}. "
            f"Vorhanden: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["x"] = df[trace_col].astype(str).str.split(TRACE_SEPARATOR).str.len()
    out["y"] = df[time_col]
    out["group"] = group_name

    return out


def main() -> None:
    df1 = _read_and_prepare(
        CSV_1_PATH, CSV_1_X_COL, CSV_1_Y_COL, CSV_1_GROUP_NAME
    )
    df2 = _read_and_prepare(
        CSV_2_PATH, CSV_2_X_COL, CSV_2_Y_COL, CSV_2_GROUP_NAME
    )

    data = pd.concat([df1, df2], ignore_index=True)

    fig = px.scatter(
        data,
        x="x",
        y="y",
        color="group",
        color_discrete_map=GROUP_COLORS,
        title= log_name + " - Berechnungszeit über Tracelänge",
        labels={
            "x": "Tracelänge (Anzahl Events)",
            "y": "Berechnungszeit",
            "group": "Gruppe",
        },
    )

    out_path = Path(OUTPUT_HTML_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


if __name__ == "__main__":
    main()
