# error: ImportError: cannot import name 'autojit'
from numba import autojit
import time

# @autojit
def foo(x, y):
    tt = time.time()
    s = 0
    for i in range(x,y):
        s += 1
    print(time.time()-tt)
    return s 

if __name__ == "__main__":
    foo(1,10000000)