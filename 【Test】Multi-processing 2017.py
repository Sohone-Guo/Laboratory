import tensorflow as tf
import numpy as np
import jieba

from multiprocessing import Process,Pool
import multiprocessing
import time
from threading import Thread

def read_file(file,splited=False):
    """ Read txt file
    
    - Arguments
        - file: str
            The address
    
    - Return
        - raw_data: str or list(if splited is True)
    
    """
    with open(file,"r",encoding="utf-8") as r:
        raw_data = r.read()
    if splited:raw_data = raw_data.split("\n")
    return raw_data

def cut(sentence):
    total = []
    for item in sentence:
        total.append(list(jieba.cut(item)))

    return total

class MyThread(Thread):

    def __init__(self, number):
        Thread.__init__(self)
        self.number = number

    def run(self):
        # self.result = list(jieba.cut(self.number))
        self.result = cut(self.number)

    def get_result(self):
        return self.result


def cut_map(sentence):
    total = []
    for item in sentence:
        total.append(MyThread(item))
    
    for item in total:
        item.start()
    
    for item in total:
        item.join()
    
    for item in total:
        item.get_result()

    


if __name__ == '__main__':

    # start = time.time()
    data = read_file("dataset/corpus/ssa.out",splited=True)[:10000]
    total1 = cut(data)
    # word = set(jieba.cut(data))
    start = time.time()
    total1 = cut(data)
    total2 = cut(data)
    total3 = cut(data)
    end = time.time()
    print(end-start)


    start = time.time()


    total1 = MyThread(data)
    total2 = MyThread(data)
    total3 = MyThread(data)
    total1.start()
    total2.start()
    total3.start()
    total1.join()
    total2.join()
    total3.join()
    end = time.time()
    print(end-start)



    # pool = Pool(processes=4)

    # start = time.time()
    # # for x in range(2):
    # #     # re = pool.apply_async(read_file,args=("dataset/corpus/ssa.out",))
    # #     # word = set(jieba.cut(data))
    # #     re = pool.apply_async(jieba.cut,args=(data,))
    # total1 = pool.apply_async(cut,args=(data,))
    # total2 = pool.apply_async(cut,args=(data,))

    # pool.close()
    # pool.join()
    # end = time.time()
    # print(end-start)

    # print(total1.get()[1])