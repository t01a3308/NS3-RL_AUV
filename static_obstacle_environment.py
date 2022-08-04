import numpy as np
class Obstacle1:
    def __init__(self):
        self.obstacle = np.array([[3, 5, -480],
                                  [10, 10, -480]])
        self.Robstacle = np.array([0.01, 0.01], dtype=float)
        self.cylinder = np.array([[8, 8],
                                  [4, 6],
                                  [6, 2]], dtype=float)
        self.cylinderR = np.array([1.5, 1, 1.5], dtype=float)  
        self.cylinderH = np.array([8, 10, 8], dtype=float) 

        self.qgoal = np.array([12, 12, -483], dtype=float)
        self.x0 = np.array([0, 0, -483], dtype=float)
Obstacle = {"Obstacle1":Obstacle1()}