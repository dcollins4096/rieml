#!/usr/bin/env python
from importlib import reload
import maker
import sys
reload(maker)

if len(sys.argv)==1:
    n = 1
else:
    n = int(sys.argv[1])

for m in range(n):
    name = maker.maker()
    print("Running %s %d/%d"%(name, m,n))
    maker.runner(name)
    maker.ploot("tubes/%s"%name)
