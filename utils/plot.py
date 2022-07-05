import pygal
import cairosvg
from pygal.style import Style

# custom_style = Style(background="#FFFFFF", plot_background="#FFFFFF")

def lineXY(y, filename, x=None, title="", **kwargs):
    """
    Save a line chart

    Parameters:
        y (dict or list):               y values (if `dict`, serie name as key
                                        and data as value)
        filename (str or PosixPath):    name of the file or full path
        x (list):                       x values
        title (str):                    title of the chart
    """
    line_chart = pygal.XY(**kwargs)
    line_chart.title = title
    if isinstance(y, dict):
        for serie in y:
            if x is None:
                x = list(range(len(y[serie])))
            line_chart.add(serie, list(zip(x, y[serie])))
    elif isinstance(y, list):
        if x is None:
            x = list(range(len(y[0])))
        for data in y:
            line_chart.add("", list(zip(x, data)))
    else:
        raise TypeError("`y` must be a `dict` or a `list`.")
    line_chart.render_to_png(str(filename))


def stacked(x, y_stacked, filename, title="", **kwargs):
    """
    Save a stacked chart

    Parameters:
       x (list):            x values
       y_stacked (dict):    y values (serie name as key and data as value)
       filename (str):      name of the file or full path
       title (str):         title of the chart
    """
    bar_chart = pygal.StackedBar(**kwargs)
    bar_chart.title = title
    bar_chart.x_labels = map(str, x)
    for serie in y_stacked:
        bar_chart.add(serie, y_stacked[serie])
    bar_chart.render_to_png(str(filename))


def chart(x, y, title="", **kwargs):
    """
    Return a line chart

    Parameters:
        x (list):       x values
        y (dict):       y values
        title (str):    title of the chart
    """
    line_chart = pygal.Line(**kwargs)
    line_chart.title = title
    line_chart.x_labels = map(str, x)
    condition = False
    if all((len(values) for values in y.values())):
        values = tuple(y.values())
        gap = abs(values[0][-1] - values[1][-1])
        same = len(values[0]) == len(values[1])
        condition = gap < 0.1 and same
    if condition:
        for i, serie in enumerate(y):
            if i == 1:
                last_value = y[serie][-1]
                values = y[serie][:-1] + [{"value": last_value, "label": str(round(last_value, 3))}]
                line_chart.add(serie, values)
            else:
                line_chart.add(serie, y[serie])
    else:
        for serie in y:
            if len(y[serie])>0:
                last_value = y[serie][-1]
                values = y[serie][:-1] + [{"value": last_value, "label": str(round(last_value, 3))}]
                line_chart.add(serie, values)
            else:
                line_chart.add(serie, y[serie])
    return line_chart


def topng(chart, **kwargs):
    return cairosvg.svg2png(bytestring=chart.render(**kwargs))
