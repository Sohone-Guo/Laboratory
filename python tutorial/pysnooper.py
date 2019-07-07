import pysnooper
import torchsnooper #@torchsnooper.snoop()

a = 10

@pysnooper.snoop("debug.log")
def count(value):
    total = []
    a = len(total)
    for item in range(value):
        total.append(item)
    a = len(total)
    return total

def test():
    a = 1
    return a



if __name__ == "__main__":
    a = count(5)
    a = test()
