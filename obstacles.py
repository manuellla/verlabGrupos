import numpy as np
from math import cos, sin, pi
import matplotlib.patches as patches

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

def drawObs(ax, obs):
  for o in obs:
    ax.add_patch(patches.Circle((o[0], o[1]), o[2], color='k'))
