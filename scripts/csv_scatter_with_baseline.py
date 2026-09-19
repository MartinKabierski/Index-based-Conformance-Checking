from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def main():
    # ----------------- Parameter -----------------
    csv_path = Path("../output/PLOTINPUT_evaluation_result_BPI_Challenge_2012.csv")
    log_name = "BPI_Challenge_2012"

    #csv_path = Path("../output/PLOTINPUT_evaluation_result_Sepsis_Cases_-_Event_Log_100_1000.csv")
    #log_name = "Sepsis_Cases"

    sep = ","

    paare = [
        ("trace_calculation_time", "fitness_value", "IBF based"),
        ("trie_based_alignment_time", "trie_based_fitness", "Trie based"),
    ]

    # BPI 2012 baseline
    baseline_x = 62677226.37
    baseline_y = 0.998033671580096

    # sepsis baseline
    #baseline_x = 25944.73080635068
    #baseline_y = 0.973764841057066
    baseline_name = "Baseline A*"

    label_col_1 = "playout_mode"
    label_col_2 = "playout_trace_variants"
    label_col_3 = "max_checked_buckets"
    label_sep = " | "

    title = log_name
    x_title = "Runtime"
    y_title = "Fitness"
    output_html = Path("../output/plots/" + log_name + ".html")

    x_log_scale = True
    y_log_scale = False

    # ----------------- Dropdown-Filter -----------------
    dropdown_1_column = label_col_1
    dropdown_1_options = {
        "Alle": None,
        "random": {"random"},
        "gumbel": {"gumbel"},
    }

    dropdown_2_column = label_col_3
    dropdown_2_options = {
        "Alle": None,
        "0": {"0"},
        "10": {"10"},
    }

    # 3) CSV-Zeilennummer als Dropdown: ALLE EINZELN auswählbar
    rowid_column_name = "csv_row"  # künstliche Spalte (1-basiert)
    dropdown_3_enabled = True

    # SICHERHEIT: Ein Dropdown mit sehr vielen Einträgen wird riesig/langsam.
    # Setze das hoch, wenn du es wirklich willst.
    max_row_dropdown_items = 5000  # harte Obergrenze inkl. "Alle"
    # --------------------------------------------

    pio.renderers.default = "browser"
    output_html.parent.mkdir(parents=True, exist_ok=True)

    needed_cols = (
        {c for x, y, _ in paare for c in (x, y)}
        | {label_col_1, label_col_2, label_col_3}
    )
    df = pd.read_csv(csv_path, sep=sep, usecols=sorted(needed_cols))

    # Zeilennummer hinzufügen (1-basiert)
    df[rowid_column_name] = df.index.astype(int) + 1

    # -------- long format --------
    frames = []
    for x_col, y_col, group in paare:
        tmp = df[[x_col, y_col, label_col_1, label_col_2, label_col_3, rowid_column_name]].rename(
            columns={x_col: "x", y_col: "y"}
        )
        tmp["gruppe"] = group

        tmp[label_col_1] = tmp[label_col_1].astype("string").fillna("")
        tmp[label_col_2] = tmp[label_col_2].astype("string").fillna("")
        tmp[label_col_3] = tmp[label_col_3].astype("string").fillna("")
        tmp[rowid_column_name] = pd.to_numeric(tmp[rowid_column_name], errors="coerce").astype("Int64")

        tmp["label"] = (
            "row=" + tmp[rowid_column_name].astype("string").fillna("")
            + label_sep + tmp[label_col_1]
            + label_sep + tmp[label_col_2]
            + label_sep + tmp[label_col_3]
        )

        frames.append(
            tmp[["x", "y", "gruppe", "label", dropdown_1_column, dropdown_2_column, rowid_column_name]]
        )

    plot_df = pd.concat(frames, ignore_index=True)

    plot_df["x"] = pd.to_numeric(plot_df["x"], errors="coerce")
    plot_df["y"] = pd.to_numeric(plot_df["y"], errors="coerce")
    plot_df = plot_df.dropna(subset=["x", "y"])

    if x_log_scale:
        plot_df = plot_df[plot_df["x"] > 0]
    if y_log_scale:
        plot_df = plot_df[plot_df["y"] > 0]

    groups = [g for _, _, g in paare]
    n_group_traces = len(groups)

    # -------- Daten vorbereiten --------
    def per_group_from_df(dfin: pd.DataFrame):
        per_group = {}
        for g in groups:
            dfg = dfin[dfin["gruppe"] == g]
            per_group[g] = (dfg["x"].to_list(), dfg["y"].to_list(), dfg["label"].to_list())
        return per_group

    def build_data_isin(filter_col: str, options: dict):
        out = {}
        for opt, values in options.items():
            if values is None:
                out[opt] = per_group_from_df(plot_df)
            else:
                out[opt] = per_group_from_df(plot_df[plot_df[filter_col].isin(values)])
        return out

    data_filter_1 = build_data_isin(dropdown_1_column, dropdown_1_options)
    data_filter_2 = build_data_isin(dropdown_2_column, dropdown_2_options)

    # Filter 3: jede einzelne Zeile
    data_filter_3 = {}
    if dropdown_3_enabled:
        unique_rows = sorted(plot_df[rowid_column_name].dropna().astype(int).unique().tolist())
        dropdown_3_options = {"Alle": None}
        for r in unique_rows:
            dropdown_3_options[str(r)] = r

        if len(dropdown_3_options) > max_row_dropdown_items:
            raise ValueError(
                f"csv_row Dropdown hätte {len(dropdown_3_options)} Einträge (Limit {max_row_dropdown_items}). "
                f"Erhöhe max_row_dropdown_items oder nutze Bereiche statt Einzelwerte."
            )

        for opt, r in dropdown_3_options.items():
            if r is None:
                data_filter_3[opt] = per_group_from_df(plot_df)
            else:
                data_filter_3[opt] = per_group_from_df(plot_df[plot_df[rowid_column_name] == r])

    # -------- Figure --------
    fig = go.Figure()

    # initial: "Alle"
    for g in groups:
        x0, y0, l0 = data_filter_1["Alle"][g]
        fig.add_trace(
            go.Scattergl(
                x=x0,
                y=y0,
                mode="markers",
                name=g,
                customdata=l0,
                marker=dict(size=7, opacity=0.65),
                hovertemplate=(
                    f"<b>{g}</b><br>"
                    "x: %{x:.3g}<br>"
                    "y: %{y:.6f}<br>"
                    "label: %{customdata}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[baseline_x],
            y=[baseline_y],
            mode="markers+text",
            name=baseline_name,
            text=[baseline_name],
            textposition="top center",
            marker=dict(size=16, symbol="x", line=dict(width=2)),
        )
    )

    def make_buttons(per_option_data: dict):
        buttons = []
        for opt, per_group in per_option_data.items():
            buttons.append(
                dict(
                    label=opt,
                    method="restyle",
                    args=[
                        {
                            "x": [per_group[g][0] for g in groups],
                            "y": [per_group[g][1] for g in groups],
                            "customdata": [per_group[g][2] for g in groups],
                        },
                        list(range(n_group_traces)),
                    ],
                )
            )
        return buttons

    # -------- Layout: Sidebar rechts --------
    sidebar_x = 1.02
    menu_style = dict(
        bgcolor="white",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
        font=dict(size=12),
        pad=dict(l=6, r=6, t=6, b=6),
        showactive=True,
    )

    updatemenus = [
        dict(
            type="dropdown",
            x=sidebar_x,
            y=1.02,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=make_buttons(data_filter_1),
            **menu_style,
        ),
        dict(
            type="dropdown",
            x=sidebar_x,
            y=0.84,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=make_buttons(data_filter_2),
            **menu_style,
        ),
    ]
    annotations = [
        dict(
            text=f"<b>Filter {dropdown_1_column}</b>",
            x=sidebar_x,
            y=1.08,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=12),
        ),
        dict(
            text=f"<b>Filter {dropdown_2_column}</b>",
            x=sidebar_x,
            y=0.90,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=12),
        ),
    ]

    if dropdown_3_enabled:
        updatemenus.append(
            dict(
                type="dropdown",
                x=sidebar_x,
                y=0.66,
                xanchor="left",
                yanchor="top",
                direction="down",
                buttons=make_buttons(data_filter_3),
                **menu_style,
            )
        )
        annotations.append(
            dict(
                text=f"<b>Filter {rowid_column_name} (single)</b>",
                x=sidebar_x,
                y=0.72,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=dict(size=12),
            )
        )
        legend_y = 0.46
        right_margin = 380
    else:
        legend_y = 0.62
        right_margin = 360

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.02, xanchor="left"),
        font=dict(size=14),
        margin=dict(l=70, r=right_margin, t=80, b=60),
        legend=dict(orientation="v", x=sidebar_x, xanchor="left", y=legend_y, yanchor="top"),
        legend_title_text="",
        updatemenus=updatemenus,
        annotations=annotations,
    )

    fig.update_xaxes(title_text=x_title, type="log" if x_log_scale else "linear")
    fig.update_yaxes(title_text=y_title, type="log" if y_log_scale else "linear")

    fig.write_html(output_html, include_plotlyjs="cdn", auto_open=True)
    fig.show()


if __name__ == "__main__":
    main()
