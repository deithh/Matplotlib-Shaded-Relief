from __future__ import annotations
from typing import Iterable, Protocol
from math import floor
from numpy import ndarray, array, zeros, linspace, isclose

class Vectorizible3(Protocol):
    def vectorize3(self) -> ndarray:
        pass
    def vectorizeRGB(self) -> ndarray:
        pass

class RGB(Vectorizible3):
    def __init__(self, r: float = 0, g: float = 0, b: float = 0) -> None:
        self.r = r
        self.g = g
        self.b = b

    def __setattr__(self, name: str, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError
        self.__dict__[name] = float(value)


    def vectorize3(self) -> ndarray:
        return array([self.r, self.g, self.b])
    
    def vectorizeRGB(self) -> ndarray:
        return self.vectorize3()
    
    def set_color(self, r: float, g: float, b: float) -> None:
        self.r = r
        self.g = g
        self.b = b

    def toHSV(self) -> HSV:
        converted = HSV()
        Cmax = max(self.r, self.g, self.b)
        Cmin = min(self.r, self.g, self.b)
        d = Cmax - Cmin

        if d == 0:
            H = 0
        elif self.r == Cmax:
            H = 60 * (self.g - self.b)/d
        elif self.g == Cmax:
            H = 60 * (2 + (self.b - self.r)/d)
        else:
            H = 60 * (4 + (self.r - self.g)/d)

        converted.h = H if H >= 0 else H + 360

        if Cmax == 0:
            converted.s = 0
        else:
            converted.s = (Cmax - Cmin)/Cmax
        
        converted.v = Cmax

        return converted


class HSV(Vectorizible3):
    def __init__(self, h: float = 0, s: float = 0, v: float = 0) -> None:
        self.h = h
        self.s = s
        self.v = v

    @property
    def h(self) -> float:
        return self._h
    
    @property
    def s(self) -> float:
        return self._s
    
    @property
    def v(self) -> float:
        return self._v
    
    @h.setter
    def h(self, val: float) -> None:
        if val > 360 or val < 0:
            raise ValueError
        self._h = val

    @v.setter
    def v(self, val: float) -> None:
        if val > 1 or val < 0:
            raise ValueError
        self._v = val

    @s.setter
    def s(self, val: float) -> None:
        if val > 1 or val < 0:
            raise ValueError
        self._s = val

    def vectorize3(self) -> ndarray:
        return array([self.h, self.s, self.v])
    
    def vectorizeRGB(self) -> ndarray:
        converted = self.toRGB()
        return converted.vectorize3()

    def set_color(self, h: float, s: float, v: float) -> None:
        self.h = h
        self.s = s
        self.v = v

    def dimm(self, shade: float) -> None:
        if self.v>shade:
            self.v -= shade
        else:
            self.v = 0

    def light_up(self, light: float) -> None:
        if self.s<light:
            self.s = 0
        else:
            self.s-=light
    
    def toRGB(self) -> RGB:
        converted = RGB()

        if self.s == 0:
            converted.set_color(self.v, self.v, self.v)

        Hi = floor(self.h / 60)
        f = self.h/60 - Hi
        p = self.v * (1-self.s)
        q = self.v *(1 - self.s * f)
        t = self.v * (1-(self.s * (1-f)))
        match Hi:
            case 0:
                converted.set_color(self.v, t, p)
            case 1:
                converted.set_color(q, self.v, p)
            case 2:
                converted.set_color(p, self.v, t)
            case 3:
                converted.set_color(p, q, self.v)
            case 4:
                converted.set_color(t, p, self.v)
            case 5:
                converted.set_color(self.v, p, q)
            case _:
                raise ValueError
            
        return converted
    
def _lerp3RGB(x: RGB, y: RGB, amt: float) -> RGB:
    xarr = x.vectorize3()
    yarr = y.vectorize3()

    interpolation = array(xarr * amt + yarr * (1-amt))
    interpolated = RGB(*interpolation)
    return interpolated


def _lerp3HSV(x: HSV, y: HSV, amt: float) -> HSV:
    if abs(y.h - x.h) > 180:
        if y.h > x.h:
            x.h += 360
        else:
            y.h += 360

    xarr = x.vectorize3()
    yarr = y.vectorize3()

    interpolation = array(xarr * amt + yarr * (1-amt))
    interpolated = HSV(*interpolation)
    interpolated.h %= 360

    return interpolated
    
    
def lerp3(x: Vectorizible3, y: Vectorizible3, amt: float) -> Vectorizible3:
    if type(x) is not type(y):
        return TypeError
    
    if type(x) is RGB:
        return _lerp3RGB(x, y, amt) 
    
    if type(x) is HSV:
        return _lerp3HSV(x, y, amt)
    
    raise TypeError


def sample_gradient_space(vectors: Iterable[Vectorizible3], amts: Iterable[float], amt: float) -> Vectorizible3:
    if amt > amts[-1] or amt < amts[0]:
        raise ValueError
    
    colors_witch_char_point = list(zip(vectors, amts))
    for (c1, begin), (c2, end) in zip(colors_witch_char_point, colors_witch_char_point[1:]):
        if amt <= end:
            return lerp3(c1, c2, 1 - (amt-begin)/(end-begin))