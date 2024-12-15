import numpy as np


class Ray:
    def __init__(self, startPoint, rayDirection, intensity):
        self.startPoint = np.array(startPoint)
        self.direction = np.array(rayDirection)
        self.length = 10
        self.intensity = intensity

        self.lengthCollection = []
        self.objectCollection = []

    def drawRay(self, ax):
        ax.quiver(*self.startPoint, *self.direction*self.length, color='g', alpha=self.intensity, arrow_length_ratio=0.1)

        # DEBUG: print(self.intensity)
