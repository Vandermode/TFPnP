import numpy as np
import matplotlib.pyplot as plt


def seq_plot(seq, xlabel, ylabel, color='blue', save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))    
    ax.plot(np.arange(1, len(seq)+1), np.array(seq),
            'o--', markersize=10, linewidth=2, color=color)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    xticks = list(range(1, len(seq)+1, max(len(seq)//5, 1)))
    if xticks[-1] != len(seq):
        xticks.append(len(seq))

    plt.xticks(xticks, fontsize=16)

    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)
    return fig, ax


def save_img(img, path):
    # img: [C, W, H] or [C,W,H,2](complex)
    # c, w, h = img.shape
    if img.shape[0] > 3:
        img = img[0:1, ...]
    if len(img.shape) == 4:
        img = img[...,0]
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img.transpose(1,2,0)
    import imageio
    imageio.imwrite(path, img)
