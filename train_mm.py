import os
import sys

if os.environ.get('LUMO_LIB', None):
    sys.path.insert(0, os.environ.get('LUMO_LIB', None))

import importlib
from pprint import pprint
import track_mm as track
from pkgutil import iter_modules
from lumo import Params

methods = {i.name for i in list(iter_modules(track.__path__))}

if __name__ == '__main__':
    params = Params()
    params.module = None
    params.from_args()
    if params.module is None or params.module not in methods:
        print(params)
        print('--module=')
        pprint(methods)
        exit(1)
    module = importlib.import_module(f'{track.__name__}.{params.module}')
    module.main()
