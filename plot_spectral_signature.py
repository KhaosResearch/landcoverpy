import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns

def plot_dataset(dataset:pd.DataFrame):
    fig, ax = pyplot.subplots(figsize=(35,15))
    ax.set(ylim=(-3, 3))
    sns.lineplot(ax=ax, x=dataset["raster"],y=dataset["mean"], hue=dataset["class"], linewidth=3)
    ax.tick_params(labelrotation=90)

    df_classes = [group.reset_index()[['class',"raster", "mean", 'std']] for _, group in dataset.groupby('class')]
    for df in df_classes:
        ax.fill_between(x=df["raster"],y1=df["mean"]+df["std"],y2=df["mean"]-df["std"], alpha=.5, linewidth=0)
    fig.get_figure().savefig("out.png") 

if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    df = df.drop(["latitude", "longitude", "spring_product_name", "autumn_product_name", "summer_product_name"], axis=1)
    df = df[df.columns.drop(list(df.filter(regex='evi$')))]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    means = df.groupby("class").mean()
    stds = df.groupby("class").std()
    means = means.reset_index()
    stds = stds.reset_index()
    stds = pd.melt(stds, id_vars="class", value_name="std", var_name="raster")
    means = pd.melt(means, id_vars="class", value_name="mean", var_name="raster")
    stds = stds["std"]
    df = means.join(stds)
    df = df[(df["class"]=="water") | (df["class"]=="wetland")]
    plot_dataset(df)
