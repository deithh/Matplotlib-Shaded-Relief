from typing import Tuple, Callable
from utils.utils import RGB, HSV, sample_gradient_space
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize


def read_dem(path: str) -> Tuple[int, np.ndarray]:
    with open(path) as file:
        data = file.readlines()
    
    data = [i.rstrip().split() for i in data]
    img = np.array(data[1:], dtype = float)
    dist = int(data[0][-1]) / 100
    return dist, img


def gradient(amt: float, tmax: float, tmin: float) -> np.ndarray:
    green = RGB(0, 1, 0).toHSV()
    yellow = RGB(1, 1, 0).toHSV()
    red = RGB(1, 0, 0).toHSV()

    colors = [green, yellow, red]
    return sample_gradient_space(colors, np.linspace(tmin, tmax, len(colors)), amt).vectorize3()


def module(arr: np.ndarray) -> float:
    return np.sqrt(np.sum(arr ** 2))


def calc_light_vector(origin: np.ndarray, light_pos: np.ndarray) -> np.ndarray:
    return origin-light_pos


def process_lights(data: np.ndarray, light_pos: np.ndarray, dist: float) -> Tuple[np.ndarray, np.ndarray]:

    padded_data = np.pad(data, 1, mode = 'symmetric')
    shades = np.zeros(shape = data.shape, dtype= float)
    lights = np.zeros(shape = data.shape, dtype= float)

    for y, row in enumerate(padded_data[1:-1,1:-1]):
        for x, h in enumerate(row):
            az = padded_data[y, x+1]                   #[o a] calculating cross product of oa x ob
            bz = padded_data[y+1, x]                   #[b -] resulting normal vector to plane o a b

            ov = np.array([0, 0, h])
            av = np.array([0, dist, az])
            bv = np.array([-dist, 0, bz])

            normal = np.cross(av - ov, bv - ov)
            light = calc_light_vector(ov, light_pos)

            thh = sum(normal * light)/(module(normal)*module(light))

            print(normal)
            shades[y, x] = 1-np.arccos(thh)/np.pi
            lights[y, x] = 0
 

    return shades, lights

def process_cmap(data: np.ndarray, gradient_func: Callable[[float, float, float], np.ndarray]) -> np.ndarray:
    
    processed = np.zeros(shape = (*data.shape, 3))

    for y, row in enumerate(data):
        for x, elem in enumerate(row):
            processed[y, x] = gradient_func(elem, data.max(), data.min())

    return processed
    

def merge_image(shades: np.ndarray,lights, cmap: np.ndarray) -> np.ndarray:
    image = np.zeros(shape = cmap.shape)

    for y, row in enumerate(image):
        for x, _ in enumerate(row):
            new = HSV(*cmap[y,x])
            new.dimm(shades[y,x])
            new.light_up(lights[y,x])
            image[y,x] = new.vectorizeRGB()
    return image

def process_image(data: np.ndarray, gradient_func: Callable[[float, float, float], np.ndarray],
                                                light_pos: np.ndarray, dist: float) -> np.ndarray:

    cmap = process_cmap(data, gradient_func)
    shades, lights = process_lights(data, light_pos, dist)

    adjust_shades = interp1d([shades.min(), shades.max()], [0, SHADE_INTENSITY])
    shades = adjust_shades(shades)

    adjust_lights = interp1d([lights.min(), lights.max()], [0, LIGHT_INTENSITY])
    lights = adjust_lights(lights)

    image = merge_image(shades, lights, cmap)

    

    return image

SHADE_INTENSITY = .6
LIGHT_INTENSITY = 0
LIGHT_POS = np.array([-100, -100, 10000])

def main() -> None:

    dist, img = read_dem("z2/big.dem")
    img = img[:150, :150]
    image = process_image(img, gradient, LIGHT_POS, dist)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()