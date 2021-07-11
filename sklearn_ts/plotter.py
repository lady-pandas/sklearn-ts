from matplotlib import pyplot as plt


def pretty_plot(chart, save=False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    chart(ax)
    ax.get_legend().remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        fig = ax.get_figure()
        fig.savefig(f'{chart.__name__}.png')
