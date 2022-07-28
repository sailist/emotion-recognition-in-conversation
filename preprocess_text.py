"""

34s [546220   3412   3412   3412]
0.58s [9359   56   56   56]
"""
import os
import sys

if os.environ.get('LUMO_LIB', None):
    sys.path.insert(0, os.environ.get('LUMO_LIB', None))

from preprocess.lexical.extractor import main

if __name__ == '__main__':
    main()
