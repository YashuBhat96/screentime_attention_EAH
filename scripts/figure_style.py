"""Choose Times New Roman for figures, or a comparable installed serif font."""

import warnings

from matplotlib import rcParams, font_manager


def use_manuscript_font():
    """Select one available serif font to avoid repeated missing font warnings."""
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=[family]),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        if family != "Times New Roman":
            warnings.warn(
                f"Times New Roman is unavailable; using {family} for figures.",
                stacklevel=2,
            )
        rcParams["font.family"] = [family]
        return
    warnings.warn("No preferred serif font found; using Matplotlib default.", stacklevel=2)
