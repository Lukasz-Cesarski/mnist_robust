import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_images(images, labels, rows=3, cols=4):
    assert len(images) == len(labels)
    assert rows * cols >= len(images)

    row_col_iterator = list(np.ndindex(rows, cols))

    titles = ["Label={}".format(label) for label in labels]
    fig = make_subplots(rows=rows,
                        cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.08
                        )

    for (row, col), image, label in zip(row_col_iterator, images, labels):
        fig.add_trace(
            go.Heatmap(
                z=image, colorscale='Greys', showscale=False),
            row=row + 1,
            col=col + 1
        )
    axes_custom_dict = dict(showline=True,
                            linewidth=2,
                            linecolor='black',
                            mirror=True,
                            ticks="",
                            fixedrange=True,
                            showticklabels=False)

    fig.update_xaxes(**axes_custom_dict)
    fig.update_yaxes(**axes_custom_dict, scaleanchor="x", scaleratio=1, autorange="reversed")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        width=800,
    )
    fig.show()
