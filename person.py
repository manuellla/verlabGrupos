import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi, arctan2, exp

PERSON_RADIUS = 15
personal_space = 0.50
GRID_SIZE = 50

class Person:
    def __init__(self, x=0, y=0, th=0,id_node=None):
        self.x = x
        self.y = y
        self.th = th
        self._radius = PERSON_RADIUS
        self.id_node = id_node
        
    def get_coords(self):
        return [self.x, self.y, self.th]

    def draw(self, draw_personal_space=False):        
        if draw_personal_space:
            self._draw_personal_space()
        self._draw_body_and_orientation()

    def _draw_body_and_orientation(self):
        body = plt.Circle((self.x, self.y), radius=self._radius, fill=False)
        plt.gca().add_patch(body)

        x_aux = self.x + self._radius * cos(self.th)
        y_aux = self.y + self._radius * sin(self.th)
        heading = plt.Line2D((self.x, x_aux), (self.y, y_aux), lw=3, color='k')
        plt.gca().add_line(heading)

    def _draw_personal_space(self):
        grid_x = np.linspace(self.x-5, self.x+5, GRID_SIZE)
        grid_y = np.linspace(self.y-5, self.y+5, GRID_SIZE)
        X, Y = np.meshgrid(grid_x, grid_y)

        Z = self._calculate_personal_space(X, Y)
        plt.contour(X, Y, Z, 10)

        return X, Y, Z

    def _calculate_personal_space(self, x, y):
        sigma_h = 2.0
        sigma_r = 1.0
        sigma_s = 4/3
        
        alpha = np.arctan2(self.y - y, self.x - x) - self.th - pi/2
        nalpha = np.arctan2(sin(alpha), cos(alpha))
        
        sigma = np.where(nalpha <= 0, sigma_r, sigma_h)

        a = (cos(self.th)**2)/(2*sigma**2) + (sin(self.th)**2)/(2*sigma_s**2)
        b = sin(2*self.th)/(4*sigma**2) - sin(2*self.th)/(4*sigma_s**2)
        c = (sin(self.th)**2)/(2*sigma**2) + (cos(self.th)**2)/(2*sigma_s**2)

        z = np.exp(-(a*(x - self.x)**2 + 2*b*(x - self.x)*(y - self.y) + c*(y - self.y)**2))
        return z
    
    
    def get_parallel_point_in_zone(self, intersection_point, zone):
        # Calculate the vector connecting intersection_point to the current person's position
        vector_to_intersection = np.array([intersection_point[0] - self.x, intersection_point[1] - self.y])
        
        # Calculate the angle between the vector and the person's orientation
        angle = np.arctan2(vector_to_intersection[1], vector_to_intersection[0]) - self.th
        
        # Calculate the distance to move parallelly based on the zone
        parallel_distance = self.personal_space if zone == 'Personal' else self.personal_space * 2
        
        # Calculate the new position using the adjusted vector and parallel_distance
        new_x = self.x + parallel_distance * np.cos(angle)
        new_y = self.y + parallel_distance * np.sin(angle)
        
        return [new_x, new_y]