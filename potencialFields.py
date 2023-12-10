import matplotlib.pyplot as plt
import numpy as np
import time
from people import generate_people, drawPeople
from obstacles import generate_obstacles, drawObs

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
    n_goals = np.random.randint(1, NPEOPLE - 1)
    goals = [np.array([np.random.uniform(-WORLD, WORLD), np.random.uniform(-WORLD, WORLD)]) for _ in range(n_goals)]
    for i in range(NPEOPLE):
        goal_idx = np.random.choice(len(goals))
        q[i] = np.append(q[i][:2], goal_idx)

# Inicialização
NPEOPLE = 10
WORLD   = 300.0
R       = 30  # Sensor range
goals = [np.array([0, 0]), np.array([0, 0])]

q = generate_people(NPEOPLE, WORLD)

k = 0.03
max_force = 2.0
goal = np.array([0, 0])

plt.close('all')
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.grid()

drawPeople(ax, q, goals, R)
plt.axis((-WORLD, WORLD, -WORLD, WORLD))

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.waitforbuttonpress()

t = 0
last_goal_change_time = time.time()
start_time = time.time()

# Loop principal
while True:
    ax.cla()
    plt.grid()

    current_time = time.time()
    if current_time - last_goal_change_time >= 30:
        random_person = np.random.randint(0, NPEOPLE)
        new_goal_idx = np.random.choice(len(goals))
        q[random_person][2] = new_goal_idx
        last_goal_change_time = current_time

    for goal in goals:
        ax.plot(goal[0], goal[1], "*", markersize=15)

    for i in range(NPEOPLE):
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

    drawPeople(ax, q, goals, R)

    elapsed_time = time.time() - start_time
    plt.axis((-WORLD, WORLD, -WORLD, WORLD))
    plt.title(elapsed_time)
    plt.draw()
    plt.pause(0.1)
    t += 1
