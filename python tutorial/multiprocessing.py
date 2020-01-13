# ----------------------------------------
# method 1
# ----------------------------------------

import time
from deco import concurrent, synchronized

class total:
    data = []

@concurrent
def test(index, total):
    # print("This is the counter of time")
    time.sleep(index)
    print("finished")

@synchronized
def call():
    a = total
    for item in range(3):
        print("This is {}".format(item))
        test(10, a)
        
    print(total)
        
# ----------------------------------------
# method 2
# ----------------------------------------

from multiprocessing import Pool

def job(num):
    return num * 2

if __name__ == '__main__':
    p = Pool(processes=20)
    data = p.map(job, [i for i in range(20)])
    p.close()
    print(data)
    from multiprocessing import Process, Lock

# ----------------------------------------
# method 3
# ----------------------------------------

def f(l, i):
    l.acquire()
    print 'hello world', i
    l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()

