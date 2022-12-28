import json
import numpy as np
import os
from tqdm import tqdm
from time import sleep

def partition(trainingData):
    return None

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
                curShot.update({'YendLoc' : event['shot']['end_location'][1], 
                'ZendLoc' : event['shot']['end_location'][-1]})
                
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

def main():
    shotMatrix, isGoal = formatData()
    pred = linReg(shotMatrix, isGoal)
    print(np.absolute(np.subtract(pred, isGoal)))
    print(np.mean(np.absolute(np.subtract(pred, isGoal))))

# main()