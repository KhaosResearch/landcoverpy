import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns

def plot_dataset(dataset:pd.DataFrame):
    fig, ax = pyplot.subplots(figsize=(15,15))
    sns.lineplot(ax=ax, x=dataset["raster"],y=dataset["mean"], hue=dataset["class"])
    ax.tick_params(labelrotation=90)
    fig.get_figure().savefig("out.png") 

if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")
    df = df[df.columns.drop(list(df.filter(regex='bri')))]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    means = df.groupby("class").mean()
    stds = df.groupby("class").std()
    means = means.reset_index()
    stds = stds.reset_index()
    stds = pd.melt(stds, id_vars="class", value_name="std", var_name="raster")
    means = pd.melt(means, id_vars="class", value_name="mean", var_name="raster")
    stds = stds["std"]
    df = means.join(stds)
    print(df)
    plot_dataset(df)