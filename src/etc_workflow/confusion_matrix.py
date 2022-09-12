import string
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib.collections import QuadMesh
from sklearn.metrics import confusion_matrix
from os.path import join
from etc_workflow.config import settings


def _write_cells(
    array_df, lin, col, o_text, facecolors, posi, fz, fmt, show_null_values=0
):

    """
        Config cell text and color.
    """

    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if cell_val != 0:
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            else:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ["%.2f%%" % per_ok, "100%"][per_ok == 100]

        # text to DEL
        text_del.append(o_text)

        # text to ADD
        font_prop = fm.FontProperties(weight="bold", size=fz)
        text_kwargs = dict(
            color="black",
            ha="center",
            va="center",
            gid="sum",
            fontproperties=font_prop,
        )
        lis_txt = ["%d" % cell_val, per_ok_s, "%.2f%%" % per_err]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic["color"] = "g"
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic["color"] = "r"
        lis_kwa.append(dic)
        lis_pos = [
            (o_text._x, o_text._y - 0.3),
            (o_text._x, o_text._y),
            (o_text._x, o_text._y + 0.3),
        ]
        for i in range(len(lis_txt)):
            new_text = dict(
                x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i],
            )
            text_add.append(new_text)

        # set background color for sum cells (last line and last column)
        # doesn't work for matplotlib==3.5.1 (ours currently)
        """carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr"""

    else:
        if per > 0:
            txt = f"$\mathbf{{{cell_val}}}$\n{per:.2f}%"
        else:
            if show_null_values == 0:
                txt = ""
            elif show_null_values == 1:
                txt = "0"
            else:
                txt = "$\mathbf{0}$\n0.0%"
        o_text.set_text(txt)

        # main diagonal
        if col == lin:
            # set color of the text in the diagonal to black
            o_text.set_color("black")
            # set background color in the diagonal to blue
            # facecolors[posi] = [0.35, 0.8, 0.55, 1.0]  # doesn't work for matplotlib==3.5.1 (ours currently)
        else:
            o_text.set_color("r")

    return text_add, text_del


def _compute_matrix(
    df_cm,
    annot=True,
    cmap="Oranges",
    fmt=".2f",
    fz=11,
    lw=2,
    cbar=False,
    figsize=(30, 30),
    show_null_values=0,
    pred_val_axis="y",
    out_image_path="./confusion_matrix.png",
):

    """
        Computes and saves the confusion matrix.

        params:
          df_cm          dataframe (pandas) without totals
          annot          print text in each cell
          cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu...
                         see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
          fz             fontsize
          lw             linewidth
          pred_val_axis  where to show the prediction values (x or y axis)
                          'col' or 'x': show predicted values in columns (x-axis) instead lines
                          'lin' or 'y': show predicted values in lines   (y-axis)
          out_image_path path where the image will be saved
    """

    if pred_val_axis in ("col", "x"):
        xlbl = "Predicted"
        ylbl = "True"
    else:
        xlbl = "True"
        ylbl = "Predicted"
        df_cm = df_cm.T

    # create "total" row/column
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm["sum_lin"] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc["sum_col"] = sum_col

    fig, ax1 = plt.subplots(figsize=figsize)

    ax = sn.heatmap(
        df_cm,
        annot=annot,
        annot_kws={"size": fz},
        linewidths=lw,
        ax=ax1,
        cbar=cbar,
        cmap=cmap,
        linecolor="black",
        fmt=fmt,
    )

    # set tick labels rotation and hide sum row/col label
    x_tick_labels = ax.get_xticklabels()
    x_tick_labels[-1] = ""
    y_tick_labels = ax.get_yticklabels()
    y_tick_labels[-1] = ""
    ax.set_xticklabels(x_tick_labels, fontsize=15)
    ax.set_yticklabels(y_tick_labels, fontsize=15)

    # face colors list
    # doesn't work for matplotlib==3.5.1 (ours currently)
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top
    for t in ax.collections[0].axes.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1

        # set text
        txt_res = _write_cells(
            array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values
        )

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    # titles and legends
    ax.set_title("Confusion matrix", fontweight="bold", fontsize=17)
    ax.set_xlabel(xlbl, fontweight="bold", fontsize=16)
    ax.set_ylabel(ylbl, fontweight="bold", fontsize=16)

    plt.savefig(out_image_path, bbox_inches="tight")


def compute_confusion_matrix(y_true, y_test, labels, out_image_path):

    """
        Create confusion matrix structure and make sure it scales properly based on the number of classes.
    """

    df_cm = pd.DataFrame(confusion_matrix(y_true, y_test))

    # Save the matrix in csv format in case it is needed (is not uploaded to MinIO)

    df_cm.columns = labels
    df_cm.index = labels

    df_cm.to_csv(out_image_path.replace(".png",".csv"),)

    df_len = len(df_cm) + 1

    _compute_matrix(df_cm, cmap="Oranges", out_image_path=out_image_path, figsize=(df_len, df_len))
    
