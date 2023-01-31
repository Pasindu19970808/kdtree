import math
from math import pi,cos,asin,sqrt
import numpy as np
def haversine_distance(lat1,lon1,lat2,lon2):
        p = pi/180
        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
        # calculate the result
        return 12742 * asin(sqrt(a))

def distance(point1:tuple,point2:tuple):
    x1,y1 = point1
    x2,y2 = point2
 
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def closest_point(all_points,new_point,distance_type):
    best_point = None
    best_dist = np.inf
    for point in all_points:
        if distance_type == "Euclidean":
            dist = distance(point,new_point)
        elif distance_type == "Haversine":
            dist = haversine_distance(point[1],point[0],new_point[1],new_point[0])
        if dist < best_dist:
            best_dist = dist
            best_point = point
    return best_dist,best_point

