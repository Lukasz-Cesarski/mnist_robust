import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def plot_images_matplotlib(images, labels, rows=3, cols=4):
    assert len(images) == len(labels)
    assert rows * cols >= len(images)
    fig, ax = plt.subplots(figsize=(16, 6), nrows=rows, ncols=cols)

    for idx, axi in enumerate(ax.flat):
        axi.imshow(images[idx], cmap='gray_r')
        axi.set_title('Digit Label: {}'.format(labels[idx]))
        axi.axis('off')
    return fig


def plot_images_plotly(images, labels, rows=3, cols=4):
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
    return fig
