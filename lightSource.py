import numpy as np
from ray import Ray


class LightSource:
    def __init__(self, rayArray, position, castDirection):
        self.position = position
        self.direction = castDirection / np.linalg.norm(castDirection)

        newRay = Ray(position, self.direction, 1)

        rayArray.append(newRay)

    def drawLightSource(self, ax):
        pos = self.position
        ax.scatter(*pos, color="r", s=30)