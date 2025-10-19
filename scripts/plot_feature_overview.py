from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import load_yaml
from src.features.engineering import build_features


def _load_price_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"], dayfirst=True)
    if "date" not in df.columns:
        raise KeyError("CSV is expected to contain a 'date' column.")
    return df.set_index("date").rename(columns=str.lower).sort_index()


def generate_feature_overview(output_path: Path, csv_path: Path) -> None:
    price_df = _load_price_dataframe(csv_path)
    features = build_features(price_df)

    # Align price/feature indexes
    common_index = features.index.intersection(price_df.index)
    price_aligned = price_df.loc[common_index]
    features_aligned = features.loc[common_index]

    # Reconstruct useful derived series from features
    rsi_cols = [c for c in features_aligned.columns if c.startswith("rsi_")]
    ma_slope_cols = [c for c in features_aligned.columns if c.startswith("ma_slope_")]
    donchian_col = "donchian_20" if "donchian_20" in features_aligned.columns else None
    macd_col = "macd_12_26" if "macd_12_26" in features_aligned.columns else None
    pct_above_ma_col = "pct_above_ma96" if "pct_above_ma96" in features_aligned.columns else None

    # Build MA levels for plotting
    ma_cols = []
    for w in (24, 48, 96):
        col_name = f"ma_{w}"
        ma_series = price_aligned["close"].rolling(w, min_periods=w).mean()
        price_aligned[col_name] = ma_series
        ma_cols.append(col_name)

    # Reconstruct ACTUAL Donchian Channel bounds (not just position indicator)
    donchian_window = 20
    donchian_high = price_aligned["close"].rolling(donchian_window, min_periods=donchian_window).max()
    donchian_low = price_aligned["close"].rolling(donchian_window, min_periods=donchian_window).min()
    donchian_mid = (donchian_high + donchian_low) / 2.0

    # Volume features
    has_volume = "volume" in price_aligned.columns
    vol_z_col = "vol_z" if "vol_z" in features_aligned.columns else None
    taker_pressure_col = "taker_pressure" if "taker_pressure" in features_aligned.columns else None

    # Volume features
    has_volume = "volume" in price_aligned.columns
    vol_z_col = "vol_z" if "vol_z" in features_aligned.columns else None
    taker_pressure_col = "taker_pressure" if "taker_pressure" in features_aligned.columns else None

    # Heatmap data (z-score for visibility)
    features_z = features_aligned.copy()
    for col in features_z.columns:
        series = features_z[col]
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            features_z[col] = 0.0
        else:
            features_z[col] = (series - series.mean()) / std
    features_z = features_z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Determine subplot layout
    # Row 1: Price + MAs + Donchian bounds
    # Row 2: RSI indicators (0-100 range)
    # Row 3: Oscillators (MACD, pct_above_ma, ma_slopes)
    # Row 4: Volume (if available)
    # Row 5: Heatmap
    rows = 5 if has_volume else 4
    row_heights = [0.35, 0.15, 0.15, 0.15, 0.20] if has_volume else [0.40, 0.15, 0.15, 0.30]

    subplot_specs = [
        [{"secondary_y": False}],  # Price panel (no secondary axis)
        [{"secondary_y": False}],  # RSI panel
        [{"secondary_y": False}],  # Oscillators panel
    ]
    if has_volume:
        subplot_specs.append([{"secondary_y": True}])  # Volume panel (taker_pressure on secondary)
    subplot_specs.append([{"type": "heatmap"}])  # Heatmap panel

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=subplot_specs,
    )

    # ============================================================
    # ROW 1: PRICE PANEL - Candlesticks + MAs + Donchian Bounds
    # ============================================================
    fig.add_trace(
        go.Candlestick(
            x=price_aligned.index,
            open=price_aligned["open"],
            high=price_aligned["high"],
            low=price_aligned["low"],
            close=price_aligned["close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )

    # Moving averages
    colors_ma = {"ma_24": "orange", "ma_48": "purple", "ma_96": "blue"}
    for col in ma_cols:
        fig.add_trace(
            go.Scatter(
                x=price_aligned.index,
                y=price_aligned[col],
                mode="lines",
                name=col.upper(),
                line=dict(width=1.5, color=colors_ma.get(col, "gray")),
            ),
            row=1,
            col=1,
        )

    # Donchian Channel bounds (upper, mid, lower)
    fig.add_trace(
        go.Scatter(
            x=price_aligned.index,
            y=donchian_high,
            mode="lines",
            name="Donchian High",
            line=dict(width=1.0, color="rgba(0,200,0,0.4)", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=price_aligned.index,
            y=donchian_mid,
            mode="lines",
            name="Donchian Mid",
            line=dict(width=1.0, color="rgba(150,150,150,0.4)", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=price_aligned.index,
            y=donchian_low,
            mode="lines",
            name="Donchian Low",
            line=dict(width=1.0, color="rgba(200,0,0,0.4)", dash="dot"),
        ),
        row=1,
        col=1,
    )

    # ============================================================
    # ROW 2: RSI PANEL (0-100 scale with reference lines)
    # ============================================================
    # Add overbought/oversold reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)

    colors_rsi = {"rsi_6": "red", "rsi_14": "blue", "rsi_21": "green"}
    for rsi_col in rsi_cols:
        fig.add_trace(
            go.Scatter(
                x=price_aligned.index,
                y=features_aligned[rsi_col],
                mode="lines",
                name=rsi_col.upper(),
                line=dict(width=1.5, color=colors_rsi.get(rsi_col, "gray")),
            ),
            row=2,
            col=1,
        )

    # ============================================================
    # ROW 3: OSCILLATORS PANEL (MACD, pct_above_ma, ma_slopes)
    # ============================================================
    # Zero line reference
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, row=3, col=1)

    if macd_col:
        fig.add_trace(
            go.Scatter(
                x=price_aligned.index,
                y=features_aligned[macd_col],
                mode="lines",
                name="MACD (12-26)",
                line=dict(width=2.0, color="blue"),
            ),
            row=3,
            col=1,
        )

    if pct_above_ma_col:
        fig.add_trace(
            go.Scatter(
                x=price_aligned.index,
                y=features_aligned[pct_above_ma_col] * 100,  # Scale to percentage
                mode="lines",
                name="% Above MA96",
                line=dict(width=1.5, color="purple"),
            ),
            row=3,
            col=1,
        )

    # MA slopes (scaled for visibility)
    colors_slope = {"ma_slope_24": "orange", "ma_slope_48": "brown", "ma_slope_96": "darkgreen"}
    for slope_col in ma_slope_cols:
        fig.add_trace(
            go.Scatter(
                x=price_aligned.index,
                y=features_aligned[slope_col] * 1000,  # Scale up for visibility
                mode="lines",
                name=f"{slope_col.upper()}×1000",
                line=dict(width=1.0, color=colors_slope.get(slope_col, "gray"), dash="dot"),
            ),
            row=3,
            col=1,
        )

    # ============================================================
    # ROW 4 (if volume): VOLUME PANEL
    # ============================================================
    heatmap_row = 5 if has_volume else 4
    if has_volume:
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=price_aligned.index,
                y=pd.to_numeric(price_aligned["volume"], errors="coerce"),
                name="Volume",
                marker_color="rgba(100,150,255,0.6)",
            ),
            row=4,
            col=1,
        )
        if "taker_buy_volume" in price_aligned.columns:
            fig.add_trace(
                go.Bar(
                    x=price_aligned.index,
                    y=pd.to_numeric(price_aligned["taker_buy_volume"], errors="coerce"),
                    name="Taker Buy Vol",
                    marker_color="rgba(200,80,120,0.6)",
                ),
                row=4,
                col=1,
            )
        
        # Taker pressure on secondary axis ([-1, 1] range)
        if taker_pressure_col:
            fig.add_trace(
                go.Scatter(
                    x=price_aligned.index,
                    y=features_aligned[taker_pressure_col],
                    mode="lines",
                    name="Taker Pressure",
                    line=dict(width=2.0, color="yellow"),
                ),
                row=4,
                col=1,
                secondary_y=True,
            )

    # ============================================================
    # ROW 5 (or 4): HEATMAP PANEL
    # ============================================================
    # Heatmap of all features
    fig.add_trace(
        go.Heatmap(
            z=features_z.T.values,
            x=features_z.index,
            y=features_z.columns,
            coloraxis="coloraxis",
            name="Feature Z-Score",
        ),
        row=heatmap_row,
        col=1,
    )

    fig.update_layout(
        title=f"Feature Overview for {csv_path.name}",
        coloraxis=dict(colorscale="RdBu", cmin=-3, cmax=3),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price (USDT)"),
        height=1400 if has_volume else 1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
    )

    # Label axes for all panels
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Oscillators", row=3, col=1)
    
    if has_volume:
        fig.update_yaxes(title_text="Volume", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Taker Pressure", range=[-1.1, 1.1], row=4, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Date", row=heatmap_row, col=1)
        fig.update_yaxes(title_text="Features", row=heatmap_row, col=1)
    else:
        fig.update_xaxes(title_text="Date", row=heatmap_row, col=1)
        fig.update_yaxes(title_text="Features", row=heatmap_row, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Feature overview saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature overview plot.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output",
        default="data/feature_overview.html",
        help="Path to output HTML file (default: data/feature_overview.html)",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    data_cfg = cfg["data"]
    csv_path = Path(data_cfg["csv_path"])
    output_path = Path(args.output)

    generate_feature_overview(output_path, csv_path)


if __name__ == "__main__":
    main()
