"""We use Times New Roman for figures to match the manuscript typography."""

import warnings

from matplotlib import rcParams, font_manager


def use_manuscript_font():
    """We use a similar serif font if Times New Roman is unavailable."""
    fonts = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError:
        warnings.warn(
            "Times New Roman is unavailable. The figures will use a serif fallback.",
            stacklevel=2,
        )
    rcParams["font.family"] = fonts
