import numpy as np
import pdb
import matplotlib as plt


def ax_plot_img(ori_img, ax, title=None,
                vmin=None, vmax=None,
                title_color=None,
                add_spines=False):
    if len(ori_img.shape) == 3:
        assert (ori_img.shape[0] in [1, 3]) or\
               (ori_img.shape[-1] in [1, 3])
        # image shape should be like (H, W, C)
        if (ori_img.shape[0] in [1, 3]):
            image = np.einsum('chw->hwc', ori_img)
        else:
            image = ori_img
        cmap = plt.colormaps['gray'] if image.shape[-1] == 1 else None
    elif len(ori_img.shape) == 2:
        image = ori_img
        cmap = plt.colormaps['gray']  # to show gray image, otherwise it will has green color
    else:
        raise NotImplementedError

    height, width = image.shape[:2]
    margin = 0
    # Keep this lim setting to keep the image in right position
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    # ax.axis('off')  # this commend will remove spines, so I use tick_params instead
    ax.tick_params(left=False, right=False, bottom=False, top=False,
                   labelbottom=False, labelleft=False,
                   labeltop=False, labelright=False)  # remove ticks & tick labels
    ax.set_aspect('equal')
    ax.set_box_aspect(1)  # ratio: height:width
    if add_spines:
        for pos_name in ['bottom', 'left', 'top', 'right']:
            ax.spines[pos_name].set_linewidth(3)
            ax.spines[pos_name].set_color('r')
    
    ax.imshow(image, aspect='auto',
              vmin=vmin, vmax=vmax,
              cmap=cmap)
    
    if title is not None:
        if title_color is not None:
            ax.set_title(title, color=title_color)
        else:
            ax.set_title(title)


def ax_plot_bar(ax, xlabel, height, bottom=0, top=None, yvals=None):
    ax.set_ylim(bottom=bottom, top=top)
    bars = ax.bar(x=xlabel, height=height)
    for id_bar, bar in enumerate(bars):
        y_pos = bar.get_height()  # Let the y_val's text show with the height of bar
        if yvals is None:
            yval = y_pos  # yval == height[id_bar]
        else:
            yval = yvals[id_bar]
        ax.text(x=bar.get_x(), y=y_pos+.005,
                s=f'{yval:.3f}', rotation=45)
    ax.set_box_aspect(1)
    ax.set_xticklabels(xlabel, rotation=30, ha='right')
    return bars


def ax_plot_heatmap(arr, ax, barlabel=None, title=None,
                    vmin=None, vmax=None, highlight_row=None,
                    s=None, cmap=None, text_color=None):
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    row, col = arr.shape
    
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    # ax.set_xticks(np.arange(arr.shape[1]+1)-.5, minor=True)
    ax.set_yticks(ticks=np.arange(row),
                  labels=np.arange(row))
    ax.set_xticks([])  # remove x_ticks
    ax.tick_params(which="minor", bottom=False, left=False)

    for id_row in range(row):
        if (highlight_row is not None) \
            and (id_row in highlight_row):
            color = "r" or text_color
            fontsize = 20.
        else:
            color = "w" or text_color
            fontsize = 15.
        
        for id_col in range(col):
            ax.text(x=id_col, y=id_row, 
                    s=s or f'{arr[id_row, id_col]:.5f}',
                    ha="center", va="center",
                    color=color, fontsize=fontsize)
    # ax.grid(visible=True, which='both', axis='both',
    #         color='w', linestyle='-', linewidth=20)
    
    ax.set_aspect((row+12)/10.0)  # y/x-scale
    # ax.set_box_aspect(row+)
    
    # cbar = ax.figure.colorbar(im, ax=ax)
    # if barlabel:
    #     cbar.ax.set_ylabel(barlabel, rotation=-90, va="bottom")
    if title:
        ax.set_title(title)
