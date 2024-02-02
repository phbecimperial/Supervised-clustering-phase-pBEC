from _collections_abc import dict_items
from numpy.random import randint, random
import math
from torch import Tensor, stack
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

def mode_nums():
    nm1 = randint(0,2)

    if nm1 == 0: nm2 = randint(0,5)
    elif nm1 == 1: nm2 = randint(0,3)
    else: nm2 = randint(0,2)

    return [nm1, nm2]

def mode_func(multi_split = 0.5):
    n = randint(1,5)
    ms = []
    for i in range(n):
        if multi_split > random() and i == 1:
            newmode = [0,0]
        else:
            newmode = mode_nums()
        
        while newmode in ms:
            newmode = mode_nums()
        ms.append(newmode)
    
    outputs = [] 
    for i in modelist:
        if i in ms:
            outputs.append(Tensor([1,0]))
        else:
            outputs.append(Tensor([0,1]))
    return ms, stack(outputs)

modelist = [
    [0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [1,0], [1,1], [1,2], [1,3], [2,0], [2,1], [2,2]
]

# modelist = []
# for i in range(0,math.factorial(6)):
#     app_list = []
#     for j in range():
#         app_list.append([j, i%4])
#     modelist.append(app_list)
# print(modelist[0:10])



# modelist = CallableDict({
#     'BEC': [[0,0]],
#     'A': [[0,0], [0,1]],
#     'B': [[0,0], [0,2]],
#     'C': [[0,0], [0,1], [0,2]],
#     'D': [[0,0], [0,1], [0,2], [1,1]],
#     'E': [[0,0], [0,3]],
#     'F': [[0,0], [0,1], [0,3]],
#     'G': [[0,0], [0,2], [0,3]],
#     'I': [[0,0], [0,1], [0,2], [0,3]],
#     'J': [[0,0], [0,1], [0,2], [0,4]],
#     'K': [[0,0], [0,1], [0,3], [2,2]],
#     'L': [[0,0], [0,1], [0,2], [0,4], [2,2]],
#     'M': [[0,0], [0,1], [0,2], [1,1], [0,3]], 
#     'Laser': laser_func
# })
