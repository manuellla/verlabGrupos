from math import cos, sin, pi
import numpy as np
from person import Person

def generate_people(n, WORLD):
    q = []
    for i in range(n):   
        # Garantindo que os robôs sejam gerados dentro dos limites
        x = np.random.uniform(-WORLD, WORLD)
        y = np.random.uniform(-WORLD, WORLD)
        goal_idx = 0 
        id_node = i 
        q.append([x, y, goal_idx, id_node])
    return np.array(q)

def drawPeople(ax, robots, goals, R):
    for robot in robots:
        x, y, goal_idx, id_node = robot 
        goal = goals[int(goal_idx)]
        th = np.arctan2(goal[1] - y, goal[0] - x)
        person = Person(x, y, th, id_node)
        person.draw(draw_personal_space=True)

