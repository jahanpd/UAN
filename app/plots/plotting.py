
import colorcet as cc
import numpy as np
# from scipy.stats.kde import gaussian_kde
import pandas as pd
from bokeh.models import BasicTicker, ColorBar,\
    LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.embed import components


def plot_heatmap(attn, probs, p, X, xcolname, ycolname):
    data = pd.DataFrame(attn*100, index=ycolname, columns=xcolname)
    prob_value = pd.DataFrame(probs, index=ycolname, columns=xcolname)
    x_value = pd.DataFrame(
        np.vstack([X for _ in range(len(ycolname))]),
        index=ycolname, columns=xcolname)
    data.columns.name = "Features"
    data.index.name = "Outcomes"
    df = pd.DataFrame(data.stack(), columns=['attention']).reset_index()
    df_prob = pd.DataFrame(prob_value.stack(), columns=['probs']).reset_index()
    df_x = pd.DataFrame(x_value.stack(), columns=['X']).reset_index()
    df['probs'] = df_prob.probs
    df['x'] = df_x.X
    colors = cc.CET_D1  # [10:-30]
    r = df.attention.abs().max()
    mapper = LinearColorMapper(palette=colors, low=-r, high=r)

    TOOLS = "hover,save"

    width = len(xcolname)
    height = len(ycolname)

    p = figure(
        title="",
        x_range=xcolname, y_range=ycolname,
        x_axis_location="above", plot_width=400 + width*30,
        plot_height=300 + height*20,
        min_border_right=200, min_border_left=200,
        tools=TOOLS, toolbar_location='below',
        tooltips=[
            ('Outcome', '@Outcomes'),
            ('Feature', '@Features'),
            ('Value', '@x'),
            ('Probability', '@probs%'),
            ('Attention', '@attention%'),
            ])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "12px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(
        x="Features", y="Outcomes", width=1, height=1,
        source=df,
        fill_color={'field': 'attention', 'transform': mapper},
        line_color=None)

    color_bar = ColorBar(
        color_mapper=mapper, major_label_text_font_size="12px",
        ticker=BasicTicker(desired_num_ticks=4),
        formatter=PrintfTickFormatter(format="%d%%"),
        label_standoff=6, border_line_color=None, location=(0, 0))

    p.add_layout(color_bar, 'right')

    script, div = components(p)

    return script, div


def plot_selfattention(attn_x, xcolname):
    # w = ((np.abs(attn)*probs) / (p.reshape(-1, 1) / 100)) * np.sign(attn)
    # weight = pd.DataFrame(w, index=ycolname, columns=xcolname)
    data = pd.DataFrame(attn_x*100, index=xcolname, columns=xcolname)
    data.columns.name = "Features"
    data.index.name = "Outcomes"
    df = pd.DataFrame(data.stack(), columns=['attention']).reset_index()
    colors = cc.CET_L17  # [10:-30]
    r = df.attention.abs().max()
    mapper = LinearColorMapper(palette=colors, low=0, high=r)

    TOOLS = "hover,save"

    width = len(xcolname)

    p = figure(
        title="",
        x_range=xcolname, y_range=xcolname,
        x_axis_location="above", plot_width=400 + width*30,
        plot_height=300 + width*20,
        min_border_right=200, min_border_left=200,
        tools=TOOLS, toolbar_location='below',
        tooltips=[
            ('Feature 1', '@Outcomes'),
            ('Feature 2', '@Features'),
            ('Attention', '@attention%'),
            ])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "12px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(
        x="Features", y="Outcomes", width=1, height=1,
        source=df,
        fill_color={'field': 'attention', 'transform': mapper},
        line_color=None)

    color_bar = ColorBar(
        color_mapper=mapper, major_label_text_font_size="12px",
        ticker=BasicTicker(desired_num_ticks=4),
        formatter=PrintfTickFormatter(format="%d%%"),
        label_standoff=6, border_line_color=None, location=(0, 0))

    p.add_layout(color_bar, 'right')

    script, div = components(p)

    return script, div
