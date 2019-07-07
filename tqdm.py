rom tqdm import tqdm,trange
from time import sleep
text = ""
for char in tqdm(["a", "b", "c", "d"]):
    text = text + char
    sleep(0.1)
