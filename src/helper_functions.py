
import os
import pandas as pd
from functools import partial

# Text size in report (points)
latex_text_size = 453

def save_latex_table(df, path=None, precision=2, formatter={}, **kwargs):
    print(
        df.style\
        .hide(axis='index')\
        .format(precision=precision, thousands=",", formatter=formatter)\
        .to_latex(
            path,
            position='H', 
            position_float='centering', 
            hrules=True,
            **kwargs
        )
    )



def display_full_frame(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        display(df)


def set_fig_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

set_report_fig_size = partial(set_fig_size, width_pt=latex_text_size)
