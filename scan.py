import os

from lumo import Params


class MyParams(Params):

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.modality = ''

