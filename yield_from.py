
def gen():
    for c in "AB":
        yield c
    for i in range(1,3):
        yield i 
        
def gen():
    yield from "AB"
    yield from range(1,3)
        
for item in gen():
    print(item)