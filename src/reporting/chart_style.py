"""Shared chart style settings for reporting visualizations."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ChartStyleConfig:
    """Defines visual defaults for report charts."""

    palette: Dict[str, str]
    font_family: str
    title_size: int
    label_size: int
    tick_size: int
    grid_style: Dict[str, object]
    axis_format: Dict[str, object]
    annotation_style: Dict[str, object]


LIGHT_CHART_STYLE = ChartStyleConfig(
    palette={
        "background": "#f8fafc",
        "axes": "#ffffff",
        "text": "#0f172a",
        "muted_text": "#475569",
        "grid": "#cbd5e1",
        "positive": "#16a34a",
        "negative": "#dc2626",
        "neutral": "#0284c7",
        "accent": "#1d4ed8",
        "highlight": "#7c3aed",
    },
    font_family="DejaVu Sans",
    title_size=14,
    label_size=11,
    tick_size=9,
    grid_style={"linestyle": "--", "linewidth": 0.8, "alpha": 0.45},
    axis_format={"percent_decimals": 1, "date_format": "%b %Y"},
    annotation_style={
        "fontsize": 9,
        "color": "#0f172a",
        "bbox": {"boxstyle": "round,pad=0.2", "fc": "#e2e8f0", "ec": "none", "alpha": 0.9},
    },
)
