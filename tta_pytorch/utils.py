from itertools import repeat
from typing import Iterable, Tuple, Union


def make_int_pair(x: Union[Tuple[int, int], int]):
    if isinstance(x, Iterable):
        if len(x) == 2:
            return tuple(x)
        elif len(x) == 1:
            return tuple(repeat(x[0], 2))
        else:
            raise ValueError(f"Invalid Iterable Input: {x}")
    elif isinstance(x, int):
        return tuple(repeat(x, 2))
    else:
        raise TypeError(f"Invald Input: {x}")
