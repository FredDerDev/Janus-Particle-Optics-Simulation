import numpy as np


class Sphere:
    def __init__(self, radius, center, goldDirection):
        self.radius = radius
        self.center = np.array(center)
        self.goldDirection = np.array(goldDirection) / np.linalg.norm(goldDirection)

    def drawSphere(self, ax):
        r = self.radius
        center = self.center
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:30j, 0.0:2.0 * pi:30j]
        x = center[0] + r * sin(phi) * cos(theta)
        y = center[1] + r * sin(phi) * sin(theta)
        z = center[2] + r * cos(phi)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
        # ax.quiver(*self.center, *self.goldDirection, color='y')
        # taken from: (McKinney: 2015)

        self.drawGoldenHalf(ax)

    def drawGoldenHalf(self, ax):

        # at the moment this is just a visualization and doesnt change with a different gold-direction
        r = self.radius + 0.01 # the small number is added so the gold looks like put on top of the PS
        center = self.center
        pi = np.pi
        cos = np.cos
        sin = np.sin
        # adjust gold position here by adding '/2' or '-' to pi
        phi, theta = np.mgrid[0.0:pi:30j, 0.0:2.0 * pi/2:30j]
        x = center[0] + r * sin(phi) * cos(theta)
        y = center[1] + r * sin(phi) * sin(theta)
        z = center[2] + r * cos(phi)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='y', alpha=0.5, linewidth=0)
        # taken from: (McKinney: 2015)

    # checks the material of the sphere at a point and returns it
    def checkMaterial(self, incomingRay):

        incomingRay = incomingRay / np.linalg.norm(incomingRay)
        angle = np.dot(incomingRay, self.goldDirection)
        print(angle)
        if angle > 0:
            material = 'Au'
        else:
            material = 'PS'
        return material
