from concurrent import futures 


def count(cc):
    return cc+1

def test(data):
    with futures.ThreadPoolExecutor(3) as executor:
        res = executor.map(count, data)
        
    return list(res)

print(test([1,2,3,4,5]))