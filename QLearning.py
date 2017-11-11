from __future__ import division
from matplotlib import pyplot as plt
import numpy as np

# Input description
rows = 4
cols = 4
start = (0,0)
goal = (3,3)
badStates = [ (1,2) ]

# Parameters
gamma = 0.99
probA = 0.9
probB = 0.05
# epsilon = 0.05
epsilon = 0.2
sigma = 1e-1

rewardGoal = 100
rewardBad = -70
rewardGeneral = -1

numEpisodes = 10000

actions = {
	"L" : (-1,0),
	"R" : (1,0),
	"U" : (0,1),
	"D" : (0,-1)
}

ACard = len(actions.keys()) 

Q = np.zeros((rows, cols, ACard))

actionIndices = {
	"L" : 0,
	"R" : 1,
	"U" : 2,
	"D" : 3
}

Nsa = np.zeros((rows, cols, ACard))

def whatHappens(action):
  happens = np.random.choice(["intended", "lateral"], p=[probA, 2*probB])
  if happens == "intended":
    a = action
  else:
    if action == "L" or action == "R":
      a = np.random.choice(["U", "D"])
    else:
      a = np.random.choice(["L", "R"])
  return a

def nextState(state, action):
  a = whatHappens(action)
  tryStateX = state[0] + actions[a][0]
  tryStateY = state[1] + actions[a][1]
  if tryStateX >= rows or tryStateX < 0 or tryStateY >= cols or tryStateY < 0:
    s = state
  else:
    s = (tryStateX, tryStateY)
  return s

def getReward(state):
  if state == goal:
    return rewardGoal
  elif state in badStates:
    return rewardBad
  else:
    return rewardGeneral

def takeAction(state):
  probabilities = np.zeros(ACard)

  aStar = np.argmax(Q[state[0]][state[1]])
  probabilities = [epsilon / ACard] * ACard
  probabilities[aStar] = 1 - epsilon + (epsilon / ACard)

  a = np.random.choice(["L", "R", "U", "D"], p=probabilities)  

  return a

def calcMax(state):
  return np.max(Q[state[0]][state[1]])

rewards = []

for episode in range(numEpisodes):
  s = start
  a = takeAction(s)
  reward = 0
  maxDeltaQ = 0

  while s != goal:

    Nsa[s[0]][s[1]][actionIndices[a]] += 1

    sDash = nextState(s,a)

    r = getReward(sDash)

    reward += r
    aDash = takeAction(sDash)

    alpha = 1.0 / Nsa[s[0]][s[1]][actionIndices[a]]

    deltaQ = alpha * (r + gamma * calcMax(sDash) - Q[s[0]][s[1]][actionIndices[a]])
    Q[s[0]][s[1]][actionIndices[a]] += deltaQ
    maxDeltaQ = max(maxDeltaQ, deltaQ)

    s = sDash
    a = aDash

  if abs(maxDeltaQ) < sigma:
    break

  rewards.append(reward)

policy = np.zeros((rows, cols))

for i in range(rows):
  for j in range(cols):
    policy[i][j] = np.argmax(Q[i][j])

print("====Policy====")
print(policy)

print

print("====Q values====")
print(Q)

x_axis = np.arange(0, len(rewards))
plt.plot(x_axis, rewards)
plt.show()