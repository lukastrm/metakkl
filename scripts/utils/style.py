from typing import Tuple

import matplotlib
import matplotlib.font_manager

FONT_FAMILY_CMU_SANS_SERIF = 'CMU Sans Serif'

Color = Tuple[float, float, float, float]


def rgb_from_int(r: int, g: int, b: int) -> Color:
    return r / 255.0, g / 255.0, b / 255.0, 1.0


def rgba_from_rgb(rgb: Color, alpha: float) -> Color:
    return rgb[0], rgb[1], rgb[2], alpha


# LaTex units
LATEX_PT_TO_INCH = 1 / 72.27

# IEEE Conference LaTeX template
LATEX_IEEE_TEXT_WIDTH = 505.89
LATEX_IEEE_COLUMN_WIDTH = 245.7181
IEEE_TEXT_WIDTH = LATEX_IEEE_TEXT_WIDTH * LATEX_PT_TO_INCH
IEEE_COLUMN_WIDTH = LATEX_IEEE_COLUMN_WIDTH * LATEX_PT_TO_INCH

# TU Berlin Design Guide Colors
TU_BERLIN_RED = rgb_from_int(196, 13, 30)
TU_BERLIN_BLACK = rgb_from_int(0, 0, 0)
TU_BERLIN_DARK_GRAY = rgb_from_int(67, 67, 67)
TU_BERLIN_LIGHT_GRAY = rgb_from_int(178, 178, 178)
TU_BERLIN_ORANGE = rgb_from_int(255, 108, 0)
TU_BERLIN_VIOLET = rgb_from_int(144, 19, 254)
TU_BERLIN_BLUE = rgb_from_int(31, 144, 204)
TU_BERLIN_GREEN = rgb_from_int(73, 203, 64)

TU_BERLIN_COLORS = (TU_BERLIN_RED, TU_BERLIN_ORANGE, TU_BERLIN_VIOLET, TU_BERLIN_BLUE, TU_BERLIN_GREEN)


def set_ieeeconf():
    matplotlib.rcParams['font.size'] = 9
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = 'Times'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
