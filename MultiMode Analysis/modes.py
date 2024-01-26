from _collections_abc import dict_items
from numpy.random import randint

class CallableDict(dict):
    def __getitem__(self, __key: None) -> None:
        val = super().__getitem__(__key)

        if callable(val):
            return val()
        return val
    
    def items(self) -> dict_items:
        elements = list(super().items())
        item_list = []
        for k,v in elements:
            if callable(v):
                v = v()
            item_list.append((k,v))

        return item_list


def laser_func():
    n = randint(1,5)
    ms = []
    for i in range(n):
        newmode = [randint(0,5), randint(0,5)]
        while any(ms) == newmode or newmode == [0,0]:
            newmode = [randint(0,5), randint(0,5)]
        ms.append(newmode)
    return ms


modelist = CallableDict({
    'BEC': [[0,0]],
    'A': [[0,0], [0,1]],
    'B': [[0,0], [0,2]],
    'C': [[0,0], [0,1], [0,2]],
    'D': [[0,0], [0,1], [0,2], [1,1]],
    'E': [[0,0], [0,3]],
    'F': [[0,0], [0,3]],
    'G': [[0,0], [0,2], [0,3]],
    'I': [[0,0], [0,1], [0,2], [0,3]],
    'J': [[0,0], [0,1], [0,2], [0,4]],
    'K': [[0,0], [0,1], [0,3], [2,2]],
    'L': [[0,0], [0,1], [0,2], [0,4], [2,2]],
    'M': [[0,0], [0,1], [0,2], [1,1], [0,3]],
    'Laser': laser_func
})

print(modelist.items())

for i in range(10):
    print(modelist['Laser'])