import pygal


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
