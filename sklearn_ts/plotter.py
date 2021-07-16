from matplotlib import pyplot as plt


def pretty_plot(chart, save=False, **kwargs):
    fig, ax = plt.subplots(nrows=1, ncols=1, **kwargs)

    chart(ax)
    ax.get_legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        fig = ax.get_figure()
        fig.savefig(f'{chart.__name__}.png')
