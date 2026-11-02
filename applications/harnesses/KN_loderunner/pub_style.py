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
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

DEFAULT_STYLE = "publication.mplstyle"
DEFAULT_FONT_DIR = "/usr/share/fonts/urw-base35"  # Nimbus Roman (Times metrics)


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
    return path


def add_style_args(p: argparse.ArgumentParser) -> None:
    """``--style`` and ``--font_dir`` options."""
    p.add_argument("--style", default=DEFAULT_STYLE,
                   help="Matplotlib style file (path, or a name searched in the "
                   "cwd, this dir and its parents); 'none' = matplotlib default.")
    p.add_argument("--font_dir", default=DEFAULT_FONT_DIR,
                   help="Extra .otf/.ttf fonts to register (the style's font).")
