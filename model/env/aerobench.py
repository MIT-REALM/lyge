"""Very simple module to add the aerobench package to the path"""

import sys
import os


dirs = [
    "/home/syzhang/Packages/AeroBenchVVPython/code",
]
for dir_name in dirs:
    if os.path.isdir(dir_name):
        sys.path.append(dir_name)
        break
