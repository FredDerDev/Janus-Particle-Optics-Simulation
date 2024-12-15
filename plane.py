import datetime
import numpy as np
import cv2


class Plane:

    def __init__(self, planeCenter, planeNormal, size):
        self.planeCenter = np.array(planeCenter)
        self.planeNormal = np.array(planeNormal)
        self.size = size
        self.imageScale = 3
        self.planeImage = np.zeros((size*self.imageScale,size*self.imageScale))
        self.pixels = []

    def drawPlane(self, ax):

        # DEBUG: ax.quiver(*self.planeCenter, *self.planeNormal, color='b')

        planeVector = np.cross(self.planeNormal, np.array([1, 0, 0])).astype(float)
        if np.linalg.norm(planeVector) < 1e-6:

            # calculates vector in the plane in x-direction
            planeVector = np.cross(self.planeNormal, np.array([0, 1, 0]))


        planeVector /= np.linalg.norm(planeVector)

        # creates the corners of the plane (as a square)
        corners = [self.planeCenter + self.size / 2 * (planeVector + np.cross(self.planeNormal, planeVector)),
                   self.planeCenter + self.size / 2 * (-planeVector + np.cross(self.planeNormal, planeVector)),
                   self.planeCenter + self.size / 2 * (-planeVector - np.cross(self.planeNormal, planeVector)),
                   self.planeCenter + self.size / 2 * (planeVector - np.cross(self.planeNormal, planeVector))]

        # creates plane mesh
        x = np.linspace(corners[0][0], corners[2][0], self.size) # divides the distance between x of the 1. and 3. corner into 'size' parts
        y = np.linspace(corners[0][1], corners[2][1], self.size)
        x, y = np.meshgrid(x, y) # creates a 2D-Array out of x- and y-Array

        # ax + by + cz = d --> z = (d - ax - by )/ c to calculate 2D Array of z-coordinates for every mesh point
        z = (self.planeNormal[0] * self.planeCenter[0] - self.planeNormal[0] * x - self.planeNormal[1] * y +
             self.planeNormal[1] * self.planeCenter[1]) / self.planeNormal[2] + self.planeCenter[2]

        # draws plane
        ax.plot_surface(x, y, z, color='k', alpha=0.4)

    def editImage(self, *coordinates, intensity):
        coordinates_2D = np.array(coordinates[:2]) # removes z-component

        if coordinates_2D[0] < self.size/2 and coordinates_2D[0] > -self.size/2 and coordinates_2D[1] < self.size/2 and coordinates_2D[1] > -self.size/2:
            self.pixels.append((((self.size/2 + coordinates_2D)*self.imageScale).astype(int), intensity))

    def exportImage(self):

        for coordinates_2D, intensity in self.pixels:
            self.planeImage[coordinates_2D[0], coordinates_2D[1]] += intensity*255

            # maximizes the brightness to 255
            if self.planeImage[coordinates_2D[0], coordinates_2D[1]] > 255:
                self.planeImage[coordinates_2D[0], coordinates_2D[1]] = 255

            # DEBUG: print(coordinates_2D, intensity)

        filename = f"exportImage_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        cv2.imwrite(filename, self.planeImage)
