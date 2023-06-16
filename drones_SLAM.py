
from typing import Dict, List
import numpy as np
import math
from matrix import matrix
import random
# from itertools import izip as zip


OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib
    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')

TUNE_MEASUREMENT_NOISE = 0.1
TUNE_MOTION_NOISE = 0.04

class SLAM:


    def __init__(self):
        """Initialize SLAM components here.
        """
        # refer codes from course codes
        dim = 2

        self.Xi = matrix()
        self.Xi.zero(dim, 1)

        self.omega = np.zeros([dim, dim], dtype = float)
        self.omega[0][0] = 1.0
        self.omega[1][1] = 1.0
        self.omega = matrix()
        self.omega.zero(dim, dim)
        self.omega.value[0][0] = 1.0
        self.omega.value[1][1] = 1.0

        self.trees = dict()
        self.steering = 0
        self.coordinates = None
        self.mu = None

    # Provided Functions
    def get_coordinates(self):
       
        if self.mu:
            coordinates = dict()
            for t, v in self.trees.items():
                x = self.mu[v][0]
                y = self.mu[v+1][0]
                coordinate = (x,y)
                coordinates[t] = coordinate
            x = self.mu[0][0]
            y = self.mu[1][0]
            init_loc = (x,y)
            coordinates['self'] = init_loc
        else: coordinates = self.coordinates

        return coordinates

    def process_measurements(self, measurements: Dict):
        
        tree_add = []
        for i in measurements.keys():
            if i not in self.trees: tree_add.append(i)

        # expand trees/landmarks
        if len(tree_add) != 0:
            self.expand_update_tree(tree_add)

        # refer codes from course lecture codes here
        for measure in measurements:
            data = measurements[measure]
            # Simply accumulating the steering using an instance variable starting from 0
            orientation = data['bearing'] + self.steering
            # orientation = orientation % (2.0 * math.pi) 
            orientation = self.compute_orientation(orientation) # radians
            distance = data['distance']
            measurement = self.compute_coordinate(distance, orientation)
            m = self.trees[measure]

            for b in range(2):
                self.omega.value[b][b] += 1.0 / TUNE_MEASUREMENT_NOISE
                self.omega.value[b][m+b] += -1.0 / TUNE_MEASUREMENT_NOISE
                self.omega.value[m+b][m+b] += 1.0 / TUNE_MEASUREMENT_NOISE
                self.omega.value[m+b][b] += -1.0 / TUNE_MEASUREMENT_NOISE
                self.Xi.value[b][0] += -measurement[b] / TUNE_MEASUREMENT_NOISE
                self.Xi.value[m+b][0] += measurement[b] / TUNE_MEASUREMENT_NOISE

        self.mu = self.omega.inverse() * self.Xi

    
    def compute_orientation(self, steer):
        orientation = (steer + 0.9*math.pi) % (2.0 * math.pi) - 0.9*math.pi
        return orientation
    
    def expand_update_tree(self, tree_add):
        # expand trees/landmarks
        num_trees = len(self.trees)
        num_trees_new = len(self.trees) + len(tree_add)
        dim = 2 * (1 + num_trees) #4
        dim1 = 2 * (1 + num_trees_new) #6
        l = len(self.omega.value)
        # l = self.omega.shape[0]
        dim_keep = [i for i in range(0, l)]
        idx_add = [i for i in range(dim, dim1, 2)]

        new_trees = {k: v for k, v in zip(tree_add, idx_add)}
        self.trees.update(new_trees)
        # self.Xi = np.pad(self.Xi, [(0,dim1-l), (0,0)], mode='constant', constant_values=0.)
        # self.omega = np.pad(self.omega, [(0,dim1-l), (0,dim1-l)], mode='constant', constant_values=0.)
        c = [0]
        self.omega = self.omega.expand(2 * (1 + num_trees_new), dim1, dim_keep, dim_keep)
        self.Xi = self.Xi.expand(2 * (1 + num_trees_new), 1, dim_keep, c)

    def expand_tree(self):
        dim = 2 * (1 + len(self.trees))
        dim1 = dim+2

        c = [0]
        dim_keep = [i for i in range(4, dim + 2)]
        dim_keep = [0, 1] + dim_keep #keep init
        self.Xi = self.Xi.expand(dim1, 1, dim_keep, c)
        self.omega = self.omega.expand(dim1, dim1, dim_keep, dim_keep)
    
    def compute_coordinate(self, distance, orientation):
        x = math.cos(orientation)*distance
        y = math.sin(orientation)*distance
        m = (x,y) 
        return m

    def process_movement(self, distance: float, steering: float):
       
        

        self.expand_tree()
        # Simply accumulating the steering
        orientation = self.steering + steering
        # self.steering = orientation % (2.0 * math.pi)
        self.steering = self.compute_orientation(orientation)
        l = len(self.omega.value)
        c, d = [0], [0, 1]

        motion = self.compute_coordinate(distance, self.steering)
        tk = list(range(2, l))
        # refer codes from course lecture codes here
        for b in range(4):
            self.omega.value[b][b] += 1.0 / TUNE_MOTION_NOISE
        for b in range(2):
            self.Xi.value[b][0] += -motion[b] / TUNE_MOTION_NOISE
            self.Xi.value[b+2][0] += motion[b] / TUNE_MOTION_NOISE
            self.omega.value[b][b+2] += -1.0 / TUNE_MOTION_NOISE
            self.omega.value[b+2][b] += -1.0 / TUNE_MOTION_NOISE
        
        Xi_tk = self.Xi.take(tk, c)
        e = self.Xi.take(d, c)
        a = self.omega.take(d)
        b = self.omega.take(d, tk)
        omega_tk = self.omega.take(tk)

        self.omega = omega_tk - b.transpose() * a.inverse() * b
        self.Xi = Xi_tk - b.transpose() * a.inverse() * e
        # self.omega = np.squeeze(np.asarray(omega))
        # self.Xi = np.squeeze(np.asarray(Xi))

        self.mu = self.omega.inverse() * self.Xi
        # self.mu = np.linalg.inv(self.omega) * self.Xi




class IndianaDronesPlanner:


    def __init__(self, max_distance: float, max_steering: float):
       
        self.steering = 0.
        self.orientation = 0.
        self.movement = 0.
        self.measurement = 0.
        self.all_movements = dict()
        self.slam = SLAM()
        self.extract = False
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.measure_bearing_noise = 0.03
        self.measure_distance_noise = 0.05
 
    def next_move(self, measurements: Dict, treasure_location: Dict):
       
        alpha = 0.25
        # detect treasure and extract
        treasure_x = treasure_location['x']
        treasure_y = treasure_location['y']
        treasure_type = treasure_location['type']
        treasure = (treasure_x, treasure_y)

        self.slam.process_measurements(measurements)
        coordinates = self.slam.get_coordinates()

        drone = coordinates['self']
        x,y = drone[0], drone[1]
        dx = x - treasure_x
        dy = y - treasure_y
        d = math.sqrt(dx ** 2 + dy ** 2) + random.gauss(0.0, self.measure_distance_noise)

        if d < self.max_distance:
            move = alpha
        else: move = self.max_distance*(1-alpha*0.6)
        if d < alpha and self.extract == False:
            self.extract = True
            return 'extract {} {} {}'.format(treasure_type, x, y), coordinates
        
    
        # turn around,all possibilities to crash and excude them
        # calculate as much possible steerings to avoid crash
        all_movements = dict()
        orientation_min = self.orientation - self.max_steering
        orientation_max = self.orientation + self.max_steering
        step = math.ceil(2*self.max_steering / (0.01 * alpha * self.max_steering))
        for turn in np.linspace(orientation_min, orientation_max, step):
            x1 = x + move * math.cos(turn)
            y1 = y + move * math.sin(turn)
            dx = x1 - treasure_x
            dy = y1 - treasure_y
            d = math.sqrt(dx ** 2 + dy ** 2) + random.gauss(0.0, self.measure_distance_noise)
            orientation = turn - self.orientation + self.measure_bearing_noise
            all_movements[d] = (orientation, x1, y1)
        Keys = list(all_movements.keys())
        Keys.sort()
        all_movements = {i: all_movements[i] for i in Keys}.items()
    
        self.movement = 0.
        self.steering = 0.
        crash = False

        for m in all_movements:
            crash = any((self.Euclidean((m[1][1], m[1][2]), coordinates[k]) <= v['radius'] + alpha) or\
                    (self.line_circle_intersect((m[1][1], m[1][2]), drone, coordinates[k], v['radius']) == True)\
                    for k, v in measurements.items())
            if crash == False:
                # update all_movements
                steering = m[1][0]
                x,y = drone[0], drone[1]
                dx = x - m[1][1]
                dy = y - m[1][2]
                movement = math.sqrt(dx ** 2 + dy ** 2) + random.gauss(0.0, self.measure_distance_noise)
                self.movement, self.steering = movement, steering
                break
        self.extract = False

        # Simply accumulating the steering
        orientation = self.steering + self.orientation
        self.orientation = self.compute_orientation(orientation)

        self.slam.process_movement(self.movement, self.steering)
        coordinates = self.slam.get_coordinates()
        
        return 'move {} {}'.format(self.movement, self.steering), coordinates
    
    def compute_orientation(self, steer):
        orientation = (steer + 0.9*math.pi) % (2.0 * math.pi) - 0.9*math.pi
        return orientation
    
    def line_circle_intersect(self, first_point, second_point, origin, radius):
        
        #https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
        x1, y1 = first_point
        x2, y2 = second_point
        
        ox,oy = origin
        r = radius
        x1 -= ox
        y1 -= oy
        x2 -= ox
        y2 -= oy
        a = (x2 - x1)**2 + (y2 - y1)**2
        b = 2*(x1*(x2 - x1) + y1*(y2 - y1))
        c = x1**2 + y1**2 - r**2
        disc = b**2 - 4*a*c

        if a == 0:
            if c <= 0:
                return True
            else:
                return False
        else: 

            if (disc <= 0):
                return False
            sqrtdisc = math.sqrt(disc)
            t1 = (-b + sqrtdisc)/(2*a)
            t2 = (-b - sqrtdisc)/(2*a)
            if((0 < t1 and t1 < 1) or (0 < t2 and t2 < 1)):
                return True
            return False
    
    def Euclidean(self, x, y):
        dx = y[0] - x[0]
        dy = y[1] - x[1]
        d = math.sqrt(dy**2+dx**2) + random.gauss(0.0, self.measure_distance_noise)
        return d


