import matplotlib.pyplot as plt
import numpy as np

def catstats(column_of_interest, data, ax):
    data["percent"] = [1] * len(data)
    a = data[[column_of_interest, "percent"]]
    a = a.groupby(column_of_interest)["percent"].count().to_frame()
    a.percent = ((a.percent / a.percent.sum()) * 100).apply(lambda x: round(x, 2))
    veri_sorted = a.sort_values("percent",ascending=False)
    veri_sorted.percent.plot(kind='bar',color = '#000088' , ax = ax)
    percent_of_CP = ["{}%".format(row["percent"]) for name,row in veri_sorted.iterrows()]
    for i,child in enumerate(ax.get_children()[:veri_sorted.index.size]):
         ax.text(i,child.get_bbox().y1+2,percent_of_CP[i], horizontalalignment ='center')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    title = "univariate stats & percentage for " + column_of_interest
    ax.set_title(title)
    ax.legend()

def catplot(list_of_columns, data, figsize):
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    list_like = np.array(list_of_columns)
    matrix_like = list_like.reshape((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i][j]
            column_of_interest = matrix_like[i][j]
            catstats(column_of_interest=column_of_interest, data=data, ax=ax)
