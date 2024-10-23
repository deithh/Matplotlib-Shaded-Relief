from utils.utils import RGB, HSV, sample_gradient_space
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import numpy as np

SAVE_PATH = "gradients.png"

def enable_tex() -> None:
    #set non-gui backend (problems with tex on default)
    rcParams["backend"] = "ps"
    #enable tex
    rcParams['text.usetex'] = True


def set_font_TNR():
#set font globally to times new roman
    rc('font', family='serif')
    rc('font', serif='Times New Roman')


def create_color_gradients(gradients: dict, size: int) -> plt.figure:
    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.25, right=0.95)

    for ax, data in zip(axes, gradients):
        gradient = data['func']
        name = data['name']
        # Create image with two lines and draw gradient on it
        
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)
        ax.tick_params( direction = 'in')

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.20
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=12)

        sec_xaxis = ax.secondary_xaxis('top')
        sec_xaxis.tick_params(direction = 'in')
        sec_xaxis.set_xticklabels([])

    
    return fig


def save_fig(fig: plt.figure, path: str) -> None:
    fig.savefig(path)


def gradient_rgb_bw(amt: float) -> np.ndarray:
    black = RGB(0, 0, 0)
    white = RGB(1, 1, 1)

    colors = [black, white]
    return sample_gradient_space(colors, np.linspace(0, 1, len(colors)), amt).vectorizeRGB()


def gradient_rgb_gbr_full(amt: float) -> np.ndarray:
    green = RGB(0, 1, 0)
    cyan = RGB(0, 1, 1)
    blue = RGB(0, 0, 1)
    magenta = RGB(1, 0, 1)
    red = RGB(1, 0 , 0)

    colors = [green, cyan, blue, magenta, red]
    return sample_gradient_space(colors, np.linspace(0, 1, len(colors)), amt).vectorizeRGB()


def gradient_rgb_gbr(amt: float) -> np.ndarray:
    red = RGB(1, 0, 0)
    green = RGB(0, 1, 0)
    blue = RGB(0, 0, 1)

    colors = [green, blue, red]
    return sample_gradient_space(colors, np.linspace(0, 1, len(colors)), amt).vectorizeRGB()


def gradient_rgb_wb_custom(amt: float) -> np.ndarray:
    white = RGB(1, 1, 1)
    green = RGB(0, 1, 0)
    cyan = RGB(0, 1, 1)
    blue = RGB(0, 0, 1)
    magenta = RGB(1, 0, 1)
    red = RGB(1, 0 , 0)
    yellow = RGB(1, 1, 0)
    black = RGB(0, 0, 0)

    colors = [white, green, cyan, blue, magenta, red, yellow, black]
    return sample_gradient_space(colors, np.linspace(0, 1, len(colors)), amt).vectorizeRGB()


def gradient_hsv_bw(amt: float) -> np.ndarray:
    black = RGB(0, 0, 0).toHSV()
    white = RGB(1, 1, 1).toHSV()

    colors = [black, white]
    return sample_gradient_space(colors, np.linspace(0, 1, len(colors)), amt).vectorizeRGB()


def gradient_hsv_gbr(amt: float) -> np.ndarray:
    red = RGB(1, 0, 0).toHSV()
    green = RGB(0, 1, 0).toHSV()
    blue = RGB(0, 0, 1).toHSV()

    colors = [green, blue, red]
    return sample_gradient_space(colors, np.linspace(0, 1, len(colors)), amt).vectorizeRGB()


def gradient_hsv_unknown(amt: float) -> np.ndarray:
    c1 = HSV(120, .5, 1)
    c2 = HSV(60, .5, 1)
    c3 = HSV(0, .5, 1)

    colors = [c1, c2, c3]
    return sample_gradient_space(colors, np.linspace(0, 1, len(colors)), amt).vectorizeRGB()


def gradient_hsv_custom(amt: float) -> np.ndarray:
    red = RGB(1, 0, 0).toHSV()
    blue = RGB(0, 0, 1).toHSV()

    return sample_gradient_space([blue, red]*10, np.linspace(0, 1, 20), amt).vectorizeRGB()


def main() -> None:
    enable_tex()
    set_font_TNR()

    #calc windowsize
    column_width_pt = 600     
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    gradients = (
        {
            "func": gradient_rgb_bw,
            "name": "RGB-BW"
        },
        {
            "func": gradient_rgb_gbr,
            "name": "RGB-GBR"
        },
        {
            "func": gradient_rgb_gbr_full,
            "name": "RGB-GBR-FULL"
        },
        {
            "func": gradient_rgb_wb_custom,
            "name": "RGB_WB-CUSTOM"
        },
        {
            "func": gradient_hsv_bw,
            "name": "HSV-BW"
        },
        {
            "func": gradient_hsv_gbr, 
            "name": "HSV-GBR"
        },
        {
            "func": gradient_hsv_unknown,
            "name": "HSV-UNKNOWN"
        },
        {
            "func": gradient_hsv_custom,
            "name": "HSV-CUSTOM"
        }
    )

    fig = create_color_gradients(gradients, size)
    save_fig(fig, SAVE_PATH)

if __name__ == "__main__":
    main()
