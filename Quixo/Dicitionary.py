from copy import deepcopy
import numpy as np
import random
import sys
import time
from tqdm import tqdm

def simmetry(key1, key2, value):
    simmetries_0 = []
    simmetries_1=[]
    key1= np.array(key1)
    if key2==0:
        simmetries_0.append((tuple(map(tuple, key1)), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(key1))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(key1, 2))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(key1, 3))), value))
        simmetries_0.append((tuple(map(tuple, np.fliplr(key1))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))
        
        map1= key1==0
        map2= key1==1
        key1[map1] = 1
        key1[map2] = 0
        value = (value[1], value[0], value[2])
        
        simmetries_1.append((tuple(map(tuple, key1)), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(key1))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(key1, 2))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(key1, 3))), value))
        simmetries_1.append((tuple(map(tuple, np.fliplr(key1))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))
    else:
        simmetries_1.append((tuple(map(tuple, key1)), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(key1))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(key1, 2))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(key1, 3))), value))
        simmetries_1.append((tuple(map(tuple, np.fliplr(key1))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
        simmetries_1.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))

        map1= key1==0
        map2= key1==1
        key1[map1] = 1
        key1[map2] = 0
        value = (value[1], value[0], value[2])

        simmetries_0.append((tuple(map(tuple, key1)), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(key1))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(key1, 2))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(key1, 3))), value))
        simmetries_0.append((tuple(map(tuple, np.fliplr(key1))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1)))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 2))), value))
        simmetries_0.append((tuple(map(tuple, np.rot90(np.fliplr(key1), 3))), value))


    return simmetries_0, simmetries_1         

def charge_dataset(names):
    dataset_0 = {}
    dataset_1 = {}
    i=0
    for name in names:
        with open(name, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.split("))")
                    line[0] = line[0][2:]
                    key1= line[0].split(")")
                    key1 = [key1[i].replace(", (", "(")[1:] for i in range(len(key1))]
                    key1 = tuple(tuple(map(int, key1[i].split(", "))) for i in range(len(key1)))
                    key2 = int(line[1][2])
                    value = line[1][6:-1].split(", ")
                    value = tuple(map(int, value))
                    simmetries_0, simmetries_1 = simmetry(key1, key2, value)
                    for (key1, value) in simmetries_0:
                        if key1 in dataset_0:
                            dataset_0[key1] = tuple(map(sum, zip(dataset_0[key1], value)))
                        else:
                            dataset_0[key1] = value

                    for (key1, value) in simmetries_1:
                        if key1 in dataset_1:
                            dataset_1[key1] = tuple(map(sum, zip(dataset_1[key1], value)))
                        else:
                            dataset_1[key1] = value 
                    
    return dataset_0, dataset_1


def distance(base, state):
    state = np.array(state)
    state_k= np.array(base)
    diff = np.subtract(state, state_k)
    num_diff_values = np.count_nonzero(diff)
    return num_diff_values

def hash_sum_even(key, vec):
    key = np.array(key).reshape(25)
    hash = []
    for v in vec:
        values=key[v]
        s=0
        s += np.sum(values)

        hash.append(s)
    return tuple(hash)



def diz(dataset, indicies):
    dizionaries=[]
    for index in indicies:
        diz = {}
        for key, value in dataset.items():
            hash = hash_sum_even(key, index) 
            if hash in diz:
                diz[hash][key]= value
            else:
                diz[hash] = {key: value}
        dizionaries.append(diz)
        
    return dizionaries
    

if __name__ == '__main__':
    names = ["Quixo\\dataset.txt"]
    dataset_0, dataset_1 = charge_dataset(names)
    print("Datatset charged")

    indicies = []
    for _ in range(5):
        l= random.sample(range(25), 25)
        tmp=[]
        for i in range(3):
            tmp.append(l[i*8:(i+1)*8])
        indicies.append(tmp)

    dizionaries = diz(dataset_0, indicies)

    print(dizionaries[0])

    
    print("Dizionaries created") 
    test = [np.random.randint(3, size=(5,5)) -1 for _ in range(100)]

    avg_dist=0
    start_time = time.time()

    for state in tqdm(test):
        best_dist = 25
        for key, value in dataset_0.items():
            dist = distance(key, state)
            if dist < best_dist:
                best_dist = dist
        
        avg_dist += best_dist
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Averange distance: {avg_dist/100}")
    print(f"Averange time: {execution_time/100}") 

    avg_dist=0
    start_time = time.time()

    for state in tqdm(test):
        best_dist = 25
        for diz_0, index in zip(dizionaries,indicies):
            hash = hash_sum_even(state, index) 
            try:
                diz_0[hash]
            except KeyError:
                continue

            for key, value in diz_0[hash].items():
                dist = distance(key, state)
                if dist < best_dist:
                    best_dist = dist
            
        avg_dist += best_dist


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Averange distance: {avg_dist/100}")
    print(f"Averange time: {execution_time/100}")