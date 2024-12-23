# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common Colors"""
from typing import Union

import torch
from jaxtyping import Float
from torch import Tensor

WHITE = torch.tensor([1.0, 1.0, 1.0])
BLACK = torch.tensor([0.0, 0.0, 0.0])
RED = torch.tensor([1.0, 0.0, 0.0])
GREEN = torch.tensor([0.0, 1.0, 0.0])
BLUE = torch.tensor([0.0, 0.0, 1.0])
Grey = torch.tensor([0.905, 0.905, 0.905])

COLORS_DICT = {
    "white": WHITE,
    "black": BLACK,
    "red": RED,
    "green": GREEN,
    "blue": BLUE,
    "grey": Grey,
}


def get_color(color: Union[str, list]) -> Float[Tensor, "3"]:
    """
    Args:
        Color as a string or a rgb list

    Returns:
        Parsed color
    """
    if isinstance(color, str):
        color = color.lower()
        if color not in COLORS_DICT:
            raise ValueError(f"{color} is not a valid preset color")
        return COLORS_DICT[color]
    if isinstance(color, list):
        if len(color) != 3:
            raise ValueError(f"Color should be 3 values (RGB) instead got {color}")
        return torch.tensor(color)

    raise ValueError(f"Color should be an RGB list or string, instead got {type(color)}")


def srgb_to_linear(srgb):
    linear = torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear

def linear_to_srgb(linear):
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1 / 2.4)) - 0.055
    )
    return srgb