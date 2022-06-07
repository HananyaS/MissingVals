import json
from itertools import product

with open('search_spaces/method3.json', 'r') as f:
    data = json.load(f)

# print(data)
x = list(product(*data.values()))
print(x)
print(len(x))
keys = data.keys()

for comb in x:
    kwargs = dict(zip(keys, comb))
    print(kwargs)

