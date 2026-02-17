"""Chart generation utilities for AlphaIntelligence newsletters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from .chart_style import LIGHT_CHART_STYLE


@dataclass
class ChartArtifact:
    """Metadata used when embedding chart references in markdown/email."""

    key: str
    title: str
    caption: str
    path: str


class MarketVisualizer:
    """Generate professional financial charts for newsletters."""

    def __init__(self, output_dir: str = "./data/reports/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = LIGHT_CHART_STYLE
        self._apply_theme()

    def _apply_theme(self) -> None:
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": self.style.font_family,
                "axes.facecolor": self.style.palette["axes"],
                "figure.facecolor": self.style.palette["background"],
                "axes.edgecolor": self.style.palette["grid"],
                "axes.labelcolor": self.style.palette["text"],
                "axes.titlesize": self.style.title_size,
                "axes.labelsize": self.style.label_size,
                "xtick.color": self.style.palette["muted_text"],
                "ytick.color": self.style.palette["muted_text"],
                "xtick.labelsize": self.style.tick_size,
                "ytick.labelsize": self.style.tick_size,
                "text.color": self.style.palette["text"],
            }
        )

    def _build_chart_path(self, chart_name: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d")
        return self.output_dir / f"{stamp}_{chart_name}.png"

    def generate_default_charts(
        self,
        index_perf: Dict[str, float],
        sector_perf: List[Dict],
        market_status: Dict,
    ) -> List[ChartArtifact]:
        """Build the default chart suite for each newsletter run."""
        artifacts: List[ChartArtifact] = []

        breadth_path = self.generate_market_breadth_snapshot(index_perf=index_perf, market_status=market_status)
        if breadth_path:
            artifacts.append(
                ChartArtifact(
                    key="market_breadth_snapshot",
                    title="Market Breadth & Index Performance Snapshot",
                    caption="Figure 1. Breadth metrics and benchmark index moves for the latest session.",
                    path=breadth_path,
                )
            )

        sector_path = self.generate_sector_leadership_chart(sector_data=sector_perf)
        if sector_path:
            artifacts.append(
                ChartArtifact(
                    key="sector_leadership",
                    title="Sector Leadership",
                    caption="Figure 2. Ranked sector leadership based on daily percentage performance.",
                    path=sector_path,
                )
            )

        context_path = self.generate_seasonality_context_chart(ticker="SPY")
        if context_path:
            artifacts.append(
                ChartArtifact(
                    key="seasonality_context",
                    title="Seasonality Context (Current Month vs Other Months)",
                    caption="Figure 3. Average monthly return context for SPY over the last 10 years.",
                    path=context_path,
                )
            )

        return artifacts

    def generate_market_breadth_snapshot(self, index_perf: Dict[str, float], market_status: Dict) -> str:
        """Create a combined chart for breadth indicators and index moves."""
        if not index_perf:
            return ""

        breadth = market_status.get("breadth", {}) if isinstance(market_status, dict) else {}
        ad_ratio = float(breadth.get("advance_decline_ratio") or 0)
        pct_above_200 = float(breadth.get("percent_above_200sma") or 0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor(self.style.palette["background"])

        index_names = list(index_perf.keys())
        index_values = list(index_perf.values())
        bar_colors = [self.style.palette["positive"] if v >= 0 else self.style.palette["negative"] for v in index_values]
        axes[0].bar(index_names, index_values, color=bar_colors, alpha=0.9)
        axes[0].axhline(0, color=self.style.palette["grid"], linewidth=1)
        axes[0].set_title("Index Performance (%)")
        axes[0].set_ylabel("Daily Return %")
        axes[0].grid(axis="y", color=self.style.palette["grid"], **self.style.grid_style)
        for idx, value in enumerate(index_values):
            axes[0].text(idx, value + (0.08 if value >= 0 else -0.18), f"{value:+.2f}%", ha="center", fontsize=9)

        metric_names = ["A/D Ratio", "% Above 200SMA"]
        metric_values = [ad_ratio, pct_above_200]
        metric_colors = [self.style.palette["accent"], self.style.palette["highlight"]]
        axes[1].bar(metric_names, metric_values, color=metric_colors, alpha=0.85)
        axes[1].set_title("Breadth Readings")
        axes[1].grid(axis="y", color=self.style.palette["grid"], **self.style.grid_style)
        for idx, value in enumerate(metric_values):
            suffix = "" if idx == 0 else "%"
            axes[1].text(idx, value + (0.5 if value >= 0 else -0.5), f"{value:.1f}{suffix}", ha="center", fontsize=9)

        fig.suptitle("Market Breadth / Index Snapshot", fontsize=self.style.title_size + 1, y=1.02)
        fig.tight_layout()
        output_path = self._build_chart_path("market_breadth_snapshot")
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    def generate_sector_leadership_chart(self, sector_data: List[Dict]) -> str:
        """Create ranked bar chart of sector performance."""
        if not sector_data:
            return ""

        df = pd.DataFrame(sector_data)
        if "changesPercentage" in df.columns:
            df["change"] = (
                df["changesPercentage"].astype(str).str.replace("%", "", regex=False).astype(float)
            )
        elif "change" in df.columns:
            df["change"] = pd.to_numeric(df["change"], errors="coerce")
        else:
            return ""

        df = df.dropna(subset=["change", "sector"]).sort_values("change", ascending=True)
        if df.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [self.style.palette["positive"] if v >= 0 else self.style.palette["negative"] for v in df["change"]]
        bars = ax.barh(df["sector"], df["change"], color=colors, alpha=0.9)
        ax.axvline(0, color=self.style.palette["grid"], linewidth=1)
        ax.grid(axis="x", color=self.style.palette["grid"], **self.style.grid_style)
        ax.set_title("Sector Leadership (Ranked Daily Move)")
        ax.set_xlabel("Return %")

        for bar in bars:
            width = bar.get_width()
            x_pos = width + (0.08 if width >= 0 else -0.35)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{width:+.2f}%", va="center", fontsize=9)

        fig.tight_layout()
        output_path = self._build_chart_path("sector_leadership")
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    def generate_seasonality_context_chart(self, ticker: str = "SPY") -> str:
        """Create a context chart for current month average vs overall monthly averages."""
        try:
            hist = yf.Ticker(ticker).history(period="10y", interval="1mo")
        except Exception:
            return ""

        if hist.empty:
            return ""

        monthly = hist["Close"].pct_change().dropna().to_frame("return")
        monthly["month"] = monthly.index.month
        month_means = monthly.groupby("month")["return"].mean() * 100
        if month_means.empty:
            return ""

        current_month = datetime.now().month
        labels = [datetime(2000, m, 1).strftime("%b") for m in month_means.index]
        values = month_means.values
        colors = [self.style.palette["neutral"] for _ in values]
        current_idx = list(month_means.index).index(current_month) if current_month in month_means.index else None
        if current_idx is not None:
            colors[current_idx] = self.style.palette["highlight"]

        fig, ax = plt.subplots(figsize=(11, 4.8))
        bars = ax.bar(labels, values, color=colors, alpha=0.9)
        ax.axhline(0, color=self.style.palette["grid"], linewidth=1)
        ax.grid(axis="y", color=self.style.palette["grid"], **self.style.grid_style)
        ax.set_title(f"{ticker} Monthly Seasonality (10Y Avg Return)")
        ax.set_ylabel("Avg Monthly Return %")

        for idx, bar in enumerate(bars):
            val = values[idx]
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.1 if val >= 0 else -0.18), f"{val:+.1f}%", ha="center", fontsize=8)

        if current_idx is not None:
            current_val = values[current_idx]
            ax.annotate(
                f"Current month: {labels[current_idx]} ({current_val:+.1f}%)",
                xy=(current_idx, current_val),
                xytext=(max(current_idx - 2, 0), current_val + 0.8),
                arrowprops={"arrowstyle": "->", "color": self.style.palette["highlight"]},
                **self.style.annotation_style,
            )

        fig.tight_layout()
        output_path = self._build_chart_path("seasonality_context")
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    def generate_price_history(self, ticker: str, price_data: pd.DataFrame, signals: List[Dict] = None) -> str:
        """Generate a price history chart with signal annotations."""
        if price_data.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(price_data.index, price_data["Close"], color=self.style.palette["accent"], linewidth=1.6, label="Price")

        if "SMA_50" in price_data.columns:
            ax.plot(price_data.index, price_data["SMA_50"], color=self.style.palette["neutral"], alpha=0.7, label="50 SMA")
        if "SMA_200" in price_data.columns:
            ax.plot(price_data.index, price_data["SMA_200"], color=self.style.palette["highlight"], alpha=0.7, label="200 SMA")

        ax.set_title(f"{ticker} Technical Analysis")
        ax.grid(True, color=self.style.palette["grid"], **self.style.grid_style)
        ax.legend(frameon=False)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate(rotation=35)

        fig.tight_layout()
        output_path = self._build_chart_path(f"{ticker.lower()}_history")
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)
