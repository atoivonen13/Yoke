"""Publication matplotlib style shared by the plot scripts.

The style file (``publication.mplstyle``, kept at the repo root) is found by
name in the cwd, this directory or any parent, so a script runs from anywhere.
Its font (Nimbus Roman, Times metrics) is registered from ``DEFAULT_FONT_DIR``
first, since it is not in matplotlib's font cache by default.

    from pub_style import add_style_args, use_pub_style
    add_style_args(parser)                       # --style, --font_dir
    use_pub_style(args.style, args.font_dir)     # whole script in the style

    with pub_context(load_pub_style(...)):       # one figure only
        ...

Scripts with screen-sized figures keep their house look under ``--style none``
and switch to the style's through the ``pub_*`` helpers (each returns the
house value unless ``use_pub_style`` applied a style):

    fig = plt.figure(figsize=pub_size((12, 10.5)))   # -> two columns wide
    ax.legend(fontsize=pub_fs(8))                    # -> legend.fontsize
    ax.set_ylabel(pub_text("long label", "short"))   # -> the short one
    ax.plot(x, y, lw=rc_lw(0.8), ms=rc_ms(3))        # scaled to lines.* sizes
    fig.savefig(path, dpi=pub_dpi(130))              # -> savefig.dpi
    pub_layout(fig)                                  # tight_layout if unstyled
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

DEFAULT_STYLE = "publication.mplstyle"
DEFAULT_FONT_DIR = "/usr/share/fonts/urw-base35"  # Nimbus Roman (Times metrics)
# Filter of each of the 9 bands (ZTF g r i, LSST u g r i z y) and its slot in
# the style's color cycle: one hue per filter, shared by ZTF and LSST (slot 5,
# sky blue, is too close to slot 0 and is left out). plot_dataset_example
# uses the same mapping.
BAND_FILTERS = ("g", "r", "i", "u", "g", "r", "i", "z", "y")
FILTER_SLOT = {"u": 0, "g": 2, "r": 1, "i": 3, "z": 4, "y": 6}
_STATE = {"styled": False}


def find_style(name: str) -> str | None:
    """Path of the style file ``name``: as given, else in the cwd, this script's
    dir or any parent of it (the repo root keeps ``publication.mplstyle``)."""
    if os.path.isfile(name):
        return os.path.abspath(name)
    if os.path.isabs(name):
        return None
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def load_pub_style(style: str = DEFAULT_STYLE,
                   font_dir: str = DEFAULT_FONT_DIR) -> str | None:
    """Register the style's fonts and return the style path (None = not found)."""
    for f in sorted(glob.glob(os.path.join(font_dir, "*.otf"))
                    + glob.glob(os.path.join(font_dir, "*.ttf"))):
        font_manager.fontManager.addfont(f)
    path = find_style(style)
    if path is None:
        print(f"WARNING: style {style!r} not found (cwd, script dir or a parent); "
              "using the default style.")
        return None
    family = matplotlib.rc_params_from_file(path, use_default_template=False).get(
        "font.serif")
    if family:
        try:
            print(f"Style {path}: font {family[0]!r} -> "
                  f"{font_manager.findfont(family[0], fallback_to_default=False)}")
        except ValueError:
            print(f"WARNING: font {family[0]!r} of {path} not installed (looked "
                  f"in {font_dir} too); matplotlib falls back to its default serif.")
    return path


def pub_context(path: str | None) -> contextlib.AbstractContextManager:
    """rc context of the publication style (no-op when ``path`` is None).

    rcParams are reset to matplotlib's defaults first so no other style leaks in.
    """
    if path is None:
        return contextlib.nullcontext()
    return plt.style.context(["default", path])


def use_pub_style(style: str = DEFAULT_STYLE,
                  font_dir: str = DEFAULT_FONT_DIR) -> str | None:
    """Apply the publication style globally (for the rest of the script)."""
    path = None if style == "none" else load_pub_style(style, font_dir)
    if path is not None:
        plt.style.use(["default", path])
    _STATE["styled"] = path is not None
    return path


def styled() -> bool:
    """Whether ``use_pub_style`` applied a style."""
    return _STATE["styled"]


def style_colors() -> list:
    """The current categorical color cycle."""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def pub_size(size: tuple, cols: float = 2) -> tuple:
    """``size`` unstyled; else ``cols`` style columns wide, same aspect ratio."""
    if not styled():
        return size
    w = cols * plt.rcParams["figure.figsize"][0]
    return w, w * size[1] / size[0]


def pub_fs(size: float, key: str = "legend.fontsize") -> float:
    """Font size ``size`` unstyled; else the style's ``key`` (in points)."""
    if not styled():
        return size
    return font_manager.FontProperties(size=plt.rcParams[key]).get_size()


def rc_lw(lw: float) -> float:
    """Line width ``lw`` (given at matplotlib's default 1.5) scaled to the style's."""
    return lw * plt.rcParams["lines.linewidth"] / 1.5


def rc_ms(ms: float) -> float:
    """Marker size ``ms`` (given at matplotlib's default 6) scaled to the style's."""
    return ms * plt.rcParams["lines.markersize"] / 6.0


def pub_title(text: str, width: int = 90) -> str:
    """``text`` unstyled; else wrapped to lines of at most ``width`` characters
    (long diagnostic titles overflow the two-column width)."""
    return textwrap.fill(text, width) if styled() else text


def pub_text(house: str, short: str) -> str:
    """``house`` unstyled; else ``short`` (axis labels sized for a 2 in panel)."""
    return short if styled() else house


def pub_dpi(dpi: float) -> float | None:
    """``dpi`` unstyled; else None (= the style's savefig.dpi)."""
    return None if styled() else dpi


def pub_layout(fig: plt.Figure) -> None:
    """tight_layout unstyled (the style uses constrained layout)."""
    if not styled():
        fig.tight_layout()


def pub_colors(colors: tuple) -> tuple:
    """``colors`` unstyled; else the style's cycle, as many as ``colors``."""
    if not styled():
        return tuple(colors)
    cyc = style_colors()
    return tuple(cyc[i % len(cyc)] for i in range(len(colors)))


def pub_band_colors(colors: tuple) -> tuple:
    """The 9 band colors: ``colors`` unstyled; else one style hue per filter."""
    if not styled():
        return tuple(colors)
    cyc = style_colors()
    return tuple(cyc[FILTER_SLOT[f] % len(cyc)] for f in BAND_FILTERS)


def add_style_args(p: argparse.ArgumentParser) -> None:
    """``--style`` and ``--font_dir`` options."""
    p.add_argument("--style", default=DEFAULT_STYLE,
                   help="Matplotlib style file (path, or a name searched in the "
                   "cwd, this dir and its parents); 'none' = matplotlib default.")
    p.add_argument("--font_dir", default=DEFAULT_FONT_DIR,
                   help="Extra .otf/.ttf fonts to register (the style's font).")
