import json
import numpy as np
import os
from tqdm import tqdm
from time import sleep

# Doesn't actually train model, but formats relevant data
# TODO: This function should be renamed
def trainModel ():
    shotData = []
    isGoal = []

    # TODO: Make number of matches an input variable
    for i in tqdm(range(50)):
        filename = os.listdir('data')[i]
        with open(f'data/{filename}', 'r') as f:
            data = json.load(f)

            # TODO: Tidy up. This is pretty ugly... and bib is a bad name
            for bib in data:
                if bib['type']['name'] == 'Shot':

                    # Calculate distance to goal
                    XdistToGoal = min(abs(bib['location'][0]), 
                    abs(bib['location'][0] - 120))
                    YdistToGoal = abs(bib['location'][1] - 40)

                    if bib['shot']['body_part']['name'] == 'Head':
                        shotData.append([XdistToGoal, YdistToGoal, 1, 0, 0])
                    elif bib['shot']['body_part']['name'] == 'Left Foot':
                        shotData.append([XdistToGoal, YdistToGoal, 0, 1, 0])
                    elif bib['shot']['body_part']['name'] == 'Right Foot':
                        shotData.append([XdistToGoal, YdistToGoal, 0, 0, 1])
                    else:
                        shotData.append([XdistToGoal, YdistToGoal, 0, 0, 0])

                    if bib['shot']['outcome']['name'] == 'Goal':
                        isGoal.append(1)
                    else:
                        isGoal.append(0)
            
        sleep(2)

    shotMatrix = np.array(shotData)
    return shotMatrix, isGoal

# Function for fitting a linear model to training data
# Currently returns predictions 
def linReg(shotMatrix, isGoal):
    X = shotMatrix
    Xtran = np.transpose(X)
    inv = np.linalg.inv(np.matmul(Xtran, X))
    y = np.transpose(np.array(isGoal))

    par = np.matmul(inv, np.matmul(Xtran, y))
    print(f'pars: {par}')

    pred = np.matmul(X, par)
    pred = np.heaviside(pred - 0.5, 0.5)
    print(pred)
    return pred

# Function for testing the model
def testModel(par, number):
    filename = os.listdir('data')[number]
    with open(f'data/{filename}', 'r') as f:
            data = json.load(f)

# This should be a main function, I'm lazy
shotMatrix, isGoal = trainModel()
pred = linReg(shotMatrix, isGoal)
print(np.absolute(np.subtract(pred, isGoal)))
print(np.mean(np.absolute(np.subtract(pred, isGoal))))