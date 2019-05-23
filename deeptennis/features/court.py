import numpy as np
from typing import Optional, NamedTuple, Tuple, Iterable


class Point(NamedTuple):
    x: float
    y: float


class Line(object):

    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def solve_point(self, x: float = None, y: float = None) -> Point:

        if x is not None:
            return Point(x, self.intercept + x * self.slope)
        elif y is not None:
            return Point((y - self.intercept) / self.slope, y)
        else:
            return None

    def intersection(self, other) -> Point:
        x = (other.intercept - self.intercept) / (self.slope - other.slope)
        y = self.slope * x + self.intercept
        return Point(x, y)

    def point_distance(self, x: int, y: int) -> float:
        return abs(self.slope * x - y + self.intercept) / (self.slope ** 2 + 1 ** 2) ** 0.5

    @staticmethod
    def from_point_slope(slope: float, point: Point):
        return Line(slope, point.y - slope * point.x)

    @staticmethod
    def from_two_points(point1: Point, point2: Point):
        slope = (point2.y - point1.y) / (point2.x - point1.x)
        return Line.from_point_slope(slope, point1)

    def __repr__(self):
        return f"({self.slope}, {self.intercept})"


class Court(object):

    def __init__(self,
                 lower_serve: Line,
                 lower_base: Line,
                 right_doubles: Line,
                 right_singles: Line,
                 left_doubles: Line,
                 left_singles: Line,
                 upper_base: Optional[Line] = None,
                 upper_serve: Optional[Line] = None):
        """
        TODO: add an angle parameter. Store court as always in unrotated position, but
        have ability to transform it using the angle
        :param lower_serve:
        :param lower_base:
        :param right_doubles:
        :param right_singles:
        :param left_doubles:
        :param left_singles:
        :param upper_base:
        :param upper_serve:
        """
        self.lower_base = lower_base
        self.lower_serve = lower_serve
        self.right_doubles = right_doubles
        self.right_singles = right_singles
        self.left_doubles = left_doubles
        self.left_singles = left_singles
        _upper_base, _upper_serve = self._get_upper_lines()
        if upper_base is None:
            self.upper_base = _upper_base
        else:
            self.upper_base = upper_base
        if upper_serve is None:
            self.upper_serve = _upper_serve
        else:
            self.upper_serve = upper_serve
        lower_left = self.lower_base.intersection(self.left_doubles)
        lower_right = self.lower_base.intersection(self.right_doubles)
        self.middle_t = Line.from_point_slope(1e8, Point((lower_left.x + lower_right.x) / 2, 0))

    def lines(self) -> Iterable[Line]:
        return [self.left_doubles, self.left_singles, self.right_doubles, self.right_singles,
                self.lower_base, self.lower_serve, self.upper_base, self.upper_serve,
                self.middle_t]

    def boundary_lines(self) -> Iterable[Line]:
        return [self.left_doubles, self.right_doubles, self.upper_base, self.lower_base]

    def _get_upper_lines(self) -> Tuple[Line, Line]:
        """
        This currently assumes the court is not rotated.
        """
        x_base, y_base = self.left_doubles.intersection(self.lower_base)
        x_serve, y_serve = self.left_doubles.intersection(self.lower_serve)
        x_base_opp, _ = self.right_doubles.intersection(self.lower_base)
        w = abs(x_base - x_base_opp)
        m = self.left_doubles.slope
        # x4 = x_base + (y_serve - y_base) * 1 / m
        # x3 = x_base_opp - (y_serve - y_base) * 1 / m
        Vy = y_base - w / 2 * -m
        Vx = x_base + w / 2
        AV = np.sqrt((x_base - Vx) ** 2 + (y_base - Vy) ** 2)
        AB = np.sqrt((x_base - x_serve) ** 2 + (y_base - y_serve) ** 2)
        C1 = 78. / 60  # dimensions of a real tennis court
        C2 = 60. / 42
        l1 = (AV - AB) * AB / (AV * C1 - (AV - AB))
        l2 = (AV - AB) * AB / (AV * C2 - (AV - AB))
        theta = np.arctan(-1 / m)
        y_top_base = y_serve - l1 * np.cos(theta)
        y_top_serve = y_serve - l2 * np.cos(theta)
        return Line(0, y_top_base), Line(0, y_top_serve)

    def contains_point(self, point: Point, tol: float = 0.0):
        cond = point.y >= self.upper_base.intercept - tol and point.y <= self.lower_base.intercept + tol
        x_left_bound, _ = self.left_doubles.solve_point(y=point.y)
        x_right_bound, _ = self.right_doubles.solve_point(y=point.y)
        cond &= (point.x >= x_left_bound - tol and point.x <= x_right_bound + tol)
        cond2 = True
        # for line in self.boundary_lines():
        #     d = line.point_distance(*point)
        #     cond2 |= (d < tolerance)
        return cond and cond2


