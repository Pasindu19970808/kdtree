import pandas as pd
import numpy as np
from math import cos,sin,sqrt,pi,asin
from kdtree import dist_func
class Node:
    def __init__(self,points:np.ndarray,depth:int,dimensions:int):
        self.split_val = None
        self.right = None
        self.left = None
        self.right_node = None 
        self.left_node = None
        self.points = points
        self.depth = depth
        self.attribute_num = dimensions
        self.leaf = False
        self.do_split()
    def do_split(self):
        #randomly select the attribute to split on 
        n = len(self.points)
        # print(f"Depth is {self.depth} with n {n}")
        # print(n) 
        if n == 0:
            self.leaf = True
            return self

        axis = self.depth%self.attribute_num
        self.sorted_points = pd.DataFrame(self.points).sort_values(by = axis).to_numpy()
        # split index
        #For even cases
        if (n%2) == 0:
            #accounts for python indexing
            median_idx = int(n/2)
        else:
            median_idx = int(np.floor(n/2))
        
        self.split_point = self.sorted_points[median_idx,:]
        self.left = self.sorted_points[:median_idx,:]
        self.right = self.sorted_points[median_idx + 1:,:]
        #We dont try to account for the fact that self.right might be an empty array. Instead we pass it as an empty array and let it get recognized as a leaf in the if n==0 check
        self.left_node = Node(self.left,self.depth + 1,self.attribute_num)
        self.right_node = Node(self.right,self.depth + 1,self.attribute_num)

        return self.left_node,self.right_node
    def query_naive(self,point,distance_metric,depth = 0,best = None):
        if self.leaf == True:
            return best
        axis = depth % self.attribute_num

        #Need to ask 2 questions
        #Did we find a better result
        #What is the next branch of recurssion
        next_best = None
        next_branch = None
        if distance_metric == "Haversine":
            #check if the distance between the point and the best point
            #is less than the distance between the point and the split point

            #Note best can be None. This would be at the root of the tree. So here we set it to split point of root as default
            if best is None or dist_func.haversine_distance(point[1],point[0],best[1],best[0]) > dist_func.haversine_distance(point[1],point[0],self.split_point[1],self.split_point[0]):
                #this means this split point is closer
                next_best = self.split_point
            else:
                next_best = best
        elif distance_metric == "Euclidean":
            if best is None or dist_func.distance(point,best) > dist_func.distance(point,self.split_point):
                #this means this split point is closer
                next_best = self.split_point
            else:
                next_best = best

        #next we have to check which side of the branch to query next
        if point[axis] < self.split_point[axis]:
            next_branch = self.left_node
        else:
            next_branch = self.right_node

        #Once the 2 questions are answered, we move to next iteration of recursion
        return next_branch.query_naive(point,distance_metric=distance_metric,depth = depth + 1, best = next_best)
    def closer_distance(self,point,p1,p2,distance_metric):
        """
        Returns p1 or p2 depending on which one is closer to the pivot. 
        p1 or p2 can be None. If None that means it is infinity distance
        """
        #if both p1 and p2 are None, we just get None
        if p1 is None:
            return p2
        if p2 is None:
            return p1

        if distance_metric == "Haversine":
            d1 = dist_func.haversine_distance(point[1],point[0],p1[1],p1[0])
            d2 = dist_func.haversine_distance(point[1],point[0],p2[1],p2[0])
        elif distance_metric == "Euclidean":
            d1 = dist_func.distance(point,p1)
            d2 = dist_func.distance(point,p2)

        if d1 < d2:
            return p1
        else:
            return p2
        
    def query_better(self,point,distance_metric,depth = 0):
        """
        The naive approach can fail. You can have a query point that is 
        very close to 2 different reference points. To avoid this, at each recurssion step we need to check if there is a sibling subtree with a better answer. If we were to draw a circle of radius best, with point of interest as the center and the circle intersects another subtree, that means there might be a better answer in that subtree.
        """
        if self.leaf == True:
            return None

        axis = depth % self.attribute_num
        next_branch = None
        opposite_branch = None

        if point[axis] < self.split_point[axis]:
            next_branch = self.left_node
            opposite_branch = self.right_node
        else:
            next_branch = self.right_node
            opposite_branch = self.left_node

        if distance_metric == "Haversine":
            #First get the closest point from the next_branch. This will go down recursively in the next_branch
            #When you go down recursively like this, there will be a leaf which returns None for the query_better function, but by this time we have already identified the best split point for that side of the branch, hence the one which is returned is the point from the parent of the leaf. 
            best = self.closer_distance(point,next_branch.query_better(point,"Haversine",depth+1),self.split_point,distance_metric)
            #If the distance between point and best in next branch is actually greater than distance between point and split point, the actual best lies in the opposite branch
            if dist_func.haversine_distance(point[1],point[0],best[1],best[0]) > abs(point[axis] - self.split_point[axis]):
                best = self.closer_distance(point,opposite_branch.query_better(point,"Haversine",depth+1),self.split_point,distance_metric)

        return best
            


class kdTree:
    def __init__(self,data):
        self._data = data
    def fit(self):
        self.kdtree = Node(self._data,depth = 0,dimensions=2)
    def predict_naive(self,point):
        closest = self.kdtree.query_naive(point,"Haversine",0,None)
        return closest



# tree = kdTree(ref_pairs)
# tree.fit()
