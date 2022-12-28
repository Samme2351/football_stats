import json
import numpy as np
import os
from tqdm import tqdm
from time import sleep
from random import shuffle, randrange
import matplotlib.pyplot as plt

# Here I'll make a new function which extracts relevant data and makes new files
def retrieveData(dirPath):
    for filename in tqdm(os.listdir(dirPath)):
        with open(f'{dirPath}/{filename}', 'r') as rawData:
            data = json.load(rawData)
            Match = {}

            shots = [event for event in data if event['type']['name'] == 'Shot']
            matchID = 0
            for event in shots:
                shotNum = shots.index(event)
                curShot = {}

                # Adds data about distance to goal when shot occured
                curShot.update({'XdistToGoal' : min(abs(event['location'][0]), abs(event['location'][0] - 120)), 
                'YdistToGoal' : abs(event['location'][1] - 40)})

                # Adds data about body part used
                if event['shot']['body_part']['name'] == 'Head':
                    curShot.update({'Head' : 1, 'LeftFoot' : 0, 'RightFoot' : 0, 'OtherBodyPart' : 0})

                elif event['shot']['body_part']['name'] == 'LeftFoot':
                    curShot.update({'Head' : 0, 'LeftFoot' : 1, 'RightFoot' : 0, 'OtherBodyPart' : 0})

                elif event['shot']['body_part']['name'] == 'RightFoot':
                    curShot.update({'Head' : 0, 'LeftFoot' : 0, 'RightFoot' : 1, 'OtherBodyPart' : 0})

                else:
                    curShot.update({'Head' : 0, 'LeftFoot' : 0, 'RightFoot' : 0, 'OtherBodyPart' : 1})

                # Adds end location of shot
                curShot.update({'YendLoc' : event['shot']['end_location'][1]})
                
                if len(event['shot']['end_location']) == 3:
                    curShot.update({'ZendLoc' : event['shot']['end_location'][2]})
                else:
                    curShot.update({'ZendLoc' : 2.67/2})
                
                if event['shot']['outcome']['name'] == 'Goal':
                    curShot.update({'Goal' : 1})
                else:
                    curShot.update({'Goal' : 0})

                Match.update({shotNum : curShot})
                matchID += 1
            
            # Writes shots of current match to 
            with open(f'shot_data/{filename}_shots.json', "w") as write_file:
                json.dump(Match, write_file)

        sleep(0.1)

# Partitions the data set into training and testing data
def partition(dirPath, testShare):
    numMatches = len(os.listdir(str(dirPath)))
    partition = [path for path in os.listdir(str(dirPath))]
    shuffle(partition)
    trainData = partition[: round(numMatches * (1 - testShare))]
    testData = partition[round(numMatches * (1 - testShare)) :]
    return trainData, testData

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

# Creates matrix of shot data and vector with whether the shot was a goal or not
def prepData(dirPath, prepData):
    shotData = []
    isGoal = []
    for filename in tqdm(prepData):
        with open(f'{str(dirPath)}/{filename}', 'r') as rawData:
            data = json.load(rawData)
            for shot in data.keys():
                curShot = []
                for key in data[shot].keys():
                    curShot.append(data[shot][key])

                isGoal.append(curShot.pop())
                shotData.append(curShot)
    
    shotData = np.array(shotData)
    isGoal = np.array(isGoal)
    isGoal.shape = (isGoal.size, 1)

    return shotData, isGoal

# Fits a logistic regression model to data
def logReg(trainData, iters, learnRate):
    shotData, isGoal = prepData('shot_data', trainData)
    params = np.zeros((np.size(shotData, 1), 1))

    costHist, paramsOpt = gradDescent(shotData, isGoal, 
    params, learnRate, iters)

    plt.figure()
    plt.plot(range(len(costHist)), costHist, 'r')
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

    print(f'Optimal params are:\n{paramsOpt}')
    return paramsOpt

# Defines sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*((((-1) * y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

def gradDescent(X, y, params, learnRate, iter):
    m = len(y)
    costHist = np.zeros((iter, 1))

    for i in tqdm(range(iter)):
        params = params - (learnRate / m) * np.matmul(np.transpose(X), sigmoid(np.matmul(X, params)) - y)
        costHist[i] = cost(X, y, params)
    
    return (costHist, params)

# Function for testing the model
def testModel(params, testData):
    shotData, isGoal = prepData('shot_data', testData)
    pred = sigmoid(shotData @ params)
    avgError = np.mean(abs(pred - isGoal))
    print(f'Average error: {avgError}')

def main():
    trainData, testData = partition('shot_data', 0.2)
    params = logReg(trainData, 200, 0.005)
    testModel(params, testData)
    return params

params = main()

def visualize(params):
    shotData, isGoal = prepData(
        'shot_data', os.listdir('shot_data'))
    xg = sigmoid(shotData @ params)
    plt.scatter(shotData[:, 0], shotData[:, 1], c = xg, cmap = 'coolwarm')
    plt.show()
    return shotData, isGoal, xg

shotData, isGoal, xg = visualize(params)

while True:
    inp = input('sample [y, g, n]: ')
    if str(inp) == 'y':
        n = randrange(0, len(os.listdir('shot_data'))-1)
        print(shotData[n, :], isGoal[n], xg[n])
    elif str(inp) == 'g':
        n = randrange(0, len(os.listdir('shot_data'))-1)
        while isGoal[n] == 0:
            n += 1
        print(shotData[n, :], isGoal[n], xg[n])
    else:
        break

n = np.where(xg == max(xg))
print(shotData[n, :], isGoal[n], xg[n])