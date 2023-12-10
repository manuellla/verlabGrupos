from math import cos, sin, pi
import numpy as np
from person import Person

def generate_people(n, WORLD):
    q = []
    for i in range(n):   
        d = np.random.uniform(WORLD/2, WORLD)
        phi = np.random.uniform(0, 2*pi)
        x = d*cos(phi)
        y = d*sin(phi)
        goal_idx = 0 
        q.append([x, y, goal_idx])
    return np.array(q)

def drawPeople(ax, robots, goals, R):
    for robot in robots:
        x, y, goal_idx = robot
        goal = goals[int(goal_idx)]
        th = np.arctan2(goal[1] - y, goal[0] - x)
        person = Person(x, y, th)
        person.draw(draw_personal_space=True)
