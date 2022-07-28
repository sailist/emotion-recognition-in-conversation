"""

34s [546220   3412   3412   3412]
0.58s [9359   56   56   56]
"""
import os
import sys

if os.environ.get('LUMO_LIB', None):
    sys.path.insert(0, os.environ.get('LUMO_LIB', None))

import importlib
from pprint import pprint
from preprocess.acoustic.extractor import main

if __name__ == '__main__':
    main()

# from preprocess import SupParams
# import track_image as track
# from pkgutil import iter_modules

# methods = {i.name for i in list(iter_modules(track.__path__))}
#
# if __name__ == '__main__':
#     params = SupParams()
#     params.module = None
#     params.from_args()
#     if params.module is None or params.module not in methods:
#         print(params)
#         print('--module=')
#         pprint(methods)
#         exit(1)
#     module = importlib.import_module(f'{track.__name__}.{params.module}')
#     module.main()
