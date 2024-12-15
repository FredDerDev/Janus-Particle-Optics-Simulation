import numpy as np
import matplotlib.pyplot as plt
import math

from sphere import Sphere
from ray import Ray
from lightSource import LightSource
from plane import Plane


# NOTE: vector names are visualized in chapter 4.2


fig = plt.figure("JP Simulation", figsize=(10, 10))
ax = plt.axes(projection='3d')

# generates lists to collect relevant objects
sphereArray = []
lightSourceArray = []
rayArray = []
planeArray = []


# cycles through object-lists and draws them
def drawElements():
    for i in range(len(sphereArray)):
        sphereArray[i].drawSphere(ax)

    for i in range(len(lightSourceArray)):
        lightSourceArray[i].drawLightSource(ax)

    for i in range(len(rayArray)):
        rayArray[i].drawRay(ax)

    for i in range(len(planeArray)):
        planeArray[i].drawPlane(ax)


# manages hit-checking of rays with all spheres and planes
def traceRays():
    for i in range(len(rayArray)):
        for j in range(len(sphereArray)):
            checkHit(rayArray[i], sphereArray[j])
        for k in range(len(planeArray)):
            checkHit(rayArray[i], planeArray[k])
        applyLength(rayArray[i])


# checks hits of rays with given object
def checkHit(ray, object):

    # collects all possible lengths of objects the ray could hit
    if object.__class__.__name__ == 'Sphere':
        sphere = object

        vector_a = ray.startPoint - sphere.center  # vector_a is the support vector of the ray
        p = 2 * vector_a.dot(ray.direction)

        q = vector_a.dot(vector_a) - sphere.radius * sphere.radius

        discriminant = p*p/4 - q

        if discriminant >= 0:
            l = -p/2 - math.sqrt(discriminant)

            # ads calculated length to the rays length collection as well as the hit object to the rays object collection
            ray.lengthCollection.append(l)
            ray.objectCollection.append(sphere)


    if object.__class__.__name__ == 'Plane':
        plane = object
        vector_a = plane.planeCenter - ray.startPoint  # vector_a is the support vector of the ray
        n = plane.planeNormal
        l = 0


        denominator = ray.direction.dot(n)

        if denominator > 0:
            l = (vector_a.dot(n))/denominator


        if l > 0:
            # ads calculated length to the rays length collection as well as the hit object to the rays object collection
            ray.lengthCollection.append(l)
            ray.objectCollection.append(plane)


# applies the smallest length to the ray
def applyLength(ray):
    if len(ray.lengthCollection) > 0:
        ray.length = min(ray.lengthCollection) # takes the smallest collected length
        hitObject = ray.objectCollection[ray.lengthCollection.index(min(ray.lengthCollection))] # gets the hit object by taking the same array index as the length

        if hitObject.__class__.__name__ == 'Sphere':

            material = hitObject.checkMaterial(ray.direction+ray.startPoint) # checks the material of the point where the ray hits the sphere
            # DEBUG: print(material)
            if material == 'PS':
                # DEBUG: print("refraction started")
                refractRay(ray, hitObject, 1, 1.5)
            if material == 'Au':
                reflectRay(ray, hitObject, 1) #  at the moment there is total reflection at gold surfaces

        if hitObject.__class__.__name__ == 'Plane':
            # DEBUG: print("Ray hit plane")
            stopPoint = ray.direction.dot(ray.length)+ray.startPoint
            # ax.scatter(*stopPoint, s=ray.intensity*1000, color='y', alpha=1) # highlights the hit spot according to intensity (to help verifying exported images)

            # filters out all rays that hit the plane 'to sharp'
            cos_theta_ray_plane = hitObject.planeNormal.dot(ray.direction)
            theta_ray_plane = np.arccos(cos_theta_ray_plane)*180/np.pi
            print("ray hit plane at "+str(theta_ray_plane)+"°")

            if(theta_ray_plane<=40):
                hitObject.editImage(*stopPoint, intensity=ray.intensity)

# reflection
def reflectRay(incomingRay, hitObject, R_Schlick):
    if hitObject.__class__.__name__ == 'Sphere':
        sphere = hitObject
        vector_a = np.array(incomingRay.startPoint - sphere.center)
        vector_p = np.array(vector_a + incomingRay.length * incomingRay.direction)
        n = vector_p / np.linalg.norm(vector_p) # calculating the normal-vector by normalizing the radius-vector
        i = incomingRay.direction / np.linalg.norm(incomingRay.direction)  # normalizes incoming ray

        r = np.array(i - 2 * (i.dot(n)) * n) # calculates direction of the reflected ray

        startPoint = np.array(sphere.center + sphere.radius * n)

        reflectedIntensity = incomingRay.intensity*R_Schlick # multiplies intensity with reflectance

        reflectedRay = Ray(startPoint, r, reflectedIntensity) # creates new Ray object
        rayArray.append(reflectedRay)

        checkHit(reflectedRay, planeArray[0]) # (has to be changed for other applications, e.g. with more Spheres)
        applyLength(reflectedRay)

        # DEBUG: ax.quiver(*startPoint, *n)

# refraction
def refractRay(incomingRay, sphere, ri1, ri2):

    vector_a = np.array(incomingRay.startPoint - sphere.center)
    vector_p = np.array(vector_a + incomingRay.length*incomingRay.direction)
    n = vector_p / np.linalg.norm(vector_p)  # calculating the normal-vector by normalizing the radius-vector
    i = incomingRay.direction / np.linalg.norm(incomingRay.direction)  # normalizes incoming ray

    startPoint = np.array(sphere.center + sphere.radius * n)

    # ax.scatter(*startPoint, color='y')

    # DEBUG: normal vector
    # ax.quiver(*startPoint, *n, color='r')
    # ax.quiver(*startPoint, *-n, color='r')

    # DEBUG: ax.quiver(*startPoint-i, *i, color='k', linewidth=2)

    cos_theta_i = -n.dot(i)

    if cos_theta_i <= 0:
        n = -n
        cos_theta_i = -n.dot(i) # inverts the normal vector for negative cos-values


    print(f"cos theta i:{cos_theta_i}")

    if cos_theta_i > 1 or cos_theta_i < -1:
        print("ERROR: cos out of bounds")
        cos_theta_i = int(cos_theta_i) # compensates rounding issues that would result in undefined arccos values



    theta_i = np.arccos(cos_theta_i)
    print(f"theta i:{theta_i*180/3.1416}")

    sinquad_theta_t = ((ri1 / ri2)*(ri1 / ri2)) * (1 - cos_theta_i*cos_theta_i)

    # leftovers (might be useful):
    # critical_angle = np.arcsin(ri2/ri1)
    # print(f"critical angle:{critical_angle*180/3.1416}")

    theta_t = np.arcsin(math.sqrt(sinquad_theta_t))
    print(f"theta_t: {theta_t*180/3.1416}")

    t = (ri1 / ri2) * i + ((ri1 / ri2) * cos_theta_i - math.sqrt(1 - sinquad_theta_t)) * n

    t = t / np.linalg.norm(t)

    # DEBUG:
    # ax.quiver(*startPoint, *t)

    R_Schlick = CalculateSchlickReflectance(cos_theta_i, ri1, ri2)

    # if the ray refracts into a sphere:
    if ri1 < ri2:
        calculateIntersectionLength(startPoint, t, sphere, incomingRay.intensity, R_Schlick)
        reflectRay(incomingRay, sphere, R_Schlick)

    # if the ray leaves a sphere:
    if ri1 > ri2:
        # DEBUG: print("no inner ray")
        newRay = Ray(startPoint, t, incomingRay.intensity-incomingRay.intensity*R_Schlick)
        rayArray.append(newRay)
        checkHit(newRay, planeArray[0])
        applyLength(newRay)
        reflectInnerRay(incomingRay, sphere, R_Schlick)

# reflectance
def CalculateSchlickReflectance(cos_theta_i, ri1, ri2):
    n_1 = ri1
    n_2 = ri2


    R_0 = ((n_1-n_2)/(n_2+n_1))*((n_1-n_2)/(n_2+n_1))


    x = (1-cos_theta_i)
    R_Schlick = R_0 + (1-R_0)*x*x*x*x*x

    print(f"R: {R_Schlick}")
    return R_Schlick

# reflection on the inside of a sphere
def reflectInnerRay(incomingRay, hitObject, R_Schlick):
    if hitObject.__class__.__name__ == 'Sphere':
        sphere = hitObject
        vector_a = np.array(incomingRay.startPoint - sphere.center)
        vector_p = np.array(vector_a + incomingRay.length * incomingRay.direction)
        n = vector_p / np.linalg.norm(vector_p) # calculating the normal-vector by normalizing the radius-vector
        i = incomingRay.direction / np.linalg.norm(incomingRay.direction)  # normalizes incoming ray

        r = np.array(i - 2 * (i.dot(n)) * n) # calculates direction of the reflected ray



        startPoint = np.array(sphere.center + sphere.radius * n)
        if sphere.checkMaterial(startPoint) == 'Au':
            intensity = incomingRay.intensity # absolute reflection
        else:
            intensity = incomingRay.intensity*R_Schlick
        if incomingRay.intensity > 0.03: # filters low intensity rays to avoid infinite calculation steps
            calculateIntersectionLength(startPoint, r, sphere, intensity, R_Schlick)

        # DEBUG: ax.quiver(*startPoint, *n)


# calculates a rays length inside of a sphere
def calculateIntersectionLength(startPoint, t_ray, sphere, intensity, R_Schlick):

    vector_a = startPoint - sphere.center
    p = 2 * vector_a.dot(t_ray)
    q = vector_a.dot(vector_a) - sphere.radius * sphere.radius
    discriminant = p * p / 4 - q
    l = -p / 2 + math.sqrt(discriminant)

    transmittedIntensity = intensity - R_Schlick*intensity


    transmittingRay = Ray(startPoint, t_ray, transmittedIntensity)
    transmittingRay.length = l
    rayArray.append(transmittingRay)
    print(f"transmiting intensity: {transmittingRay.intensity}")
    transmittedStartPoint = transmittingRay.startPoint+transmittingRay.direction*transmittingRay.length

    # DEBUG: ax.scatter(*transmittedStartPoint, color='r')
    if sphere.checkMaterial(transmittedStartPoint) == 'PS':
        refractRay(transmittingRay, sphere, 1.5, 1)
    else:
        if transmittedIntensity > 0.03:
            reflectInnerRay(transmittingRay, sphere, R_Schlick)


ax.set_box_aspect([1, 1, 1])
ax.grid()

ax.plot3D(1, 1, 1)
ax.set_title('Janus-Partikel')

# set labels of axes
ax.set_xlabel('x', labelpad=5)
ax.set_ylabel('y', labelpad=5)
ax.set_zlabel('z', labelpad=5)

# set limits of axes
lim1 = 5
lim2 = -5

ax.set_xlim(lim1, lim2)
ax.set_ylim(lim1, lim2)
ax.set_zlim(lim1, lim2)

ax.view_init(elev=-180, azim=90, roll=0)

# highlight origin
ax.scatter([0], [0], [0], color='k', s=5)


# --objects can be added down below--

newSphere1 = Sphere(3, [0,0,5], [0,1,0])
sphereArray.append(newSphere1)

newPlane = Plane([0,0,10], [0,0,1], 30)
planeArray.append(newPlane)

# newSphere2 = Sphere(1, [0,0,8], [0,-1,0])
# sphereArray.append(newSphere2)


# --dark field illumination--

LS_per_circle = 49 # select the number of light sources per circle
LS_per_stack = 10 # select the number of light sources stacked on top of each other
LS_stack_distance = 1/8 # select how far the stacked light sources are apart
minimum_height = 1 # select minimum height of the light sources
circle_radius = 12 # select circle radius

for i in range(LS_per_circle):
   for h in range(LS_per_stack):

        newLighsource = LightSource(rayArray,   [circle_radius * math.sin(i * 2*np.pi / LS_per_circle), circle_radius * math.cos(i * 2*np.pi / LS_per_circle), h*LS_stack_distance+minimum_height], [-math.sin(i * 2*np.pi / LS_per_circle), -math.cos(i * 2*np.pi/ LS_per_circle), 0.45])
        lightSourceArray.append(newLighsource)


# --circular parallel illumination--

# for i in range(LS_per_circle):
 #  newLightsource = LightSource(rayArray, [1.5 * math.sin(i * 2*np.pi / LS_per_circle), 1.5 * math.cos(i * 2*np.pi / LS_per_circle), -4], [0,0,1])
  # lightSourceArray.append(newLightsource)


# --single lightsource benchmarks--

# newLighsource = LightSource(rayArray, [0,-1.7,-4],[0,0,1])
# lightSourceArray.append(newLighsource)
# newLighsource = LightSource(rayArray, [0,2,-4],[0,0,1])

# newLighsource = LightSource(rayArray, [1.5,0,0],[0,0,1])


traceRays()
drawElements()

print(f"Number of rays: {len(rayArray)-1}")

if input('Export plane image (type: "Y"):') == 'Y':
    planeArray[0].exportImage()
else:
    pass

plt.show()


