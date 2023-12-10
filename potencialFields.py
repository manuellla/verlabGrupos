from math import *
import numpy as np
from itertools import cycle
from person import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from IPython.display import clear_output

import time

def generate_obstacles(n, WORLD):

  obs = []
  for i in range(n):
    d = np.random.uniform(-WORLD, WORLD)
    phi = np.random.uniform(0, 2*pi)
    
    x = d*cos(phi)
    y = d*sin(phi)    
    r = np.random.uniform(20, 40)    
    
    obs.append([x, y, r])  
  
  return np.array(obs)

def generate_robots(n, WORLD):
    q = []
    for i in range(n):   
        d = np.random.uniform(WORLD/2, WORLD)
        phi = np.random.uniform(0, 2*pi)
        x = d*cos(phi)
        y = d*sin(phi)
        goal_idx = 0 
        q.append([x, y, goal_idx])
    return np.array(q)

def drawObs(ax, obs):
  for o in obs:
    ax.add_patch(patches.Circle((o[0], o[1]), o[2], color='k'))
	
def drawRob(ax, robots, goals, R):
    for robot in robots:
        x, y, goal_idx = robot
        goal = goals[int(goal_idx)]
        th = np.arctan2(goal[1] - y, goal[0] - x)
        person = Person(x, y, th)
        person.draw(draw_personal_space=True)

def att_force(q, goal, katt=.01):
    return katt*(goal - q)	
    
def get_rep_force(q, people, R):
  rep = [0, 0]

  for other_person in people:
    if np.array_equal(q[:2], other_person[:2]):
        continue 
    vij = q[:2] - other_person[:2]
    dij = np.linalg.norm(vij)
    
    if dij <= 2*R: 
      repulsion_magnitude = 1e6  
      rep += repulsion_magnitude * ((1/dij) - (1/(2*R))) * (1/pow(dij, 2)) * (vij/dij)

  return rep


def on_click(event):
    global goals, q
    n_goals = np.random.randint(1, NROBOTS - 1)  # número aleatório de objetivos
    goals = [np.array([np.random.uniform(-WORLD, WORLD), np.random.uniform(-WORLD, WORLD)]) for _ in range(n_goals)]
    
    # atribuindo cada pessoa a um objetivo aleatório
    for i in range(NROBOTS):
        goal_idx = np.random.choice(len(goals))
        q[i] = np.append(q[i][:2], goal_idx)
    
#####################################################################
# Initialization
#####################################################################

NROBOTS = 10
WORLD   = 300.0
R       = 30 # sensor range
goals = [np.array([0, 0]), np.array([0, 0])]

q = generate_robots(NROBOTS, WORLD) 
print(q)

k = 0.03
max_force = 2.0
goal = np.array([0, 0])

cmap = np.random.rand(NROBOTS,)

plt.close('all')
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal') 
plt.grid()

drawRob(ax, q, goals, R)

plt.axis((-WORLD , WORLD , -WORLD , WORLD))

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.waitforbuttonpress()
  
t = 0

last_goal_change_time = time.time()

start_time = time.time()

while (True):
    ax.cla()
    plt.grid()

    current_time = time.time()
    if current_time - last_goal_change_time >= 30: # a pessoa muda de objetivo a cada 30 segundos
        # escolhendo uma pessoa aleatória e um novo objetivo aleatório para ela
        random_person = np.random.randint(0, NROBOTS)
        new_goal_idx = np.random.choice(len(goals))
        q[random_person][2] = new_goal_idx

        # atualizando o tempo da última mudança de objetivo
        last_goal_change_time = current_time

    for goal in goals:
        ax.plot(goal[0], goal[1], "*", markersize=15)

    for i in range(NROBOTS):
        goal_idx = int(q[i][2])
        goal = goals[goal_idx]

        at = att_force(q[i][:2], goal, k)
        q_positions = np.array([robot[:2] for robot in q])
        rep = get_rep_force(q[i][:2], np.delete(q_positions, i, axis=0), R)
        ui = at + rep

        q[i][:2] = q[i][:2] + ui

        ax.arrow(q[i][0], q[i][1], 20*at[0], 20*at[1], color='g')
        ax.arrow(q[i][0], q[i][1], 20*rep[0], 20*rep[1], color='r')    
        ax.arrow(q[i][0], q[i][1], 20*ui[0], 20*ui[1], color='b')

    drawRob(ax, q, goals, R)

    elapsed_time = time.time() - start_time

    plt.axis((-WORLD, WORLD, -WORLD, WORLD))
    plt.title(elapsed_time)
    plt.draw()
    plt.pause(0.1)
    
    t += 1