from operator import itemgetter 

data = [("a",1,"aa"),("b","2","bb"),("c",3,"cc")]

cc_name = itemgetter(1,0)
for item in data:
    print(cc_name(item))

