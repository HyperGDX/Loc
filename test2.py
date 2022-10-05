import numpy as np
from numpy import sqrt, dot, cross
from numpy.linalg import norm


def trilaterate(P1, P2, P3, r1, r2, r3):
    temp1 = P2-P1
    e_x = temp1/norm(temp1)
    temp2 = P3-P1
    i = dot(e_x, temp2)
    temp3 = temp2 - i*e_x
    e_y = temp3/norm(temp3)
    e_z = cross(e_x, e_y)
    d = norm(P2-P1)
    j = dot(e_y, temp2)
    x = (r1*r1 - r2*r2 + d*d) / (2*d)
    y = (r1*r1 - r3*r3 - 2*i*x + i*i + j*j) / (2*j)
    temp4 = r1*r1 - x*x - y*y
    if temp4 < 0:
        raise Exception("The three spheres do not intersect!")
    z = sqrt(temp4)
    p_12_a = P1 + x*e_x + y*e_y + z*e_z
    p_12_b = P1 + x*e_x + y*e_y - z*e_z
    return p_12_a, p_12_b


beacon_loc = [(0, 0, 3), (0, 4, 3), (4, 0, 3), (4, 4, 3)]
r = [3.1261, 2.4026, 3.9715, 3.4311]
p1 = np.array([0, 0, 3])
p2 = np.array([0, 4, 3])
p3 = np.array([4, 0, 3])
p4 = np.array([4, 4, 3])
print(trilaterate(p1, p2, p3, r[0], r[1], r[2]))
