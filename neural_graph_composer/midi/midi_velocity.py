"""
Midi velocity range
Author Mus spyroot@gmail.com
"""


class VelocityRange:
    def __init__(self, vmin: int, vmax: int):
        self.min: int = vmin
        self.max: int = vmax
