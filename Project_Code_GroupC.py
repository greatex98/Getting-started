"""
BDA450 Project

In this project we simulate merging cars into highway traffic during times of high traffic volume. Our goal of the
project is to determine whether late merging is more efficient for cars entering highway via an on ramp than early
merging. We ran the simulation 5 times for only early merging cars and then 5 times for only late merging cars. The
output of the simulation is modified such that statistics from the warm up period (i.e. the time before there is a
high traffic volume) are discarded. Then the mean, standard deviation, and confidence intervals are calculated for the
average speed (i.e. speed of traffic flow), waiting times, and throughput. In addition, histograms of average speed
and waiting times are created.

Authors: Gabrielle McCabe, Anushree Nadkarni, Jean Leclerc, and Hyunsu Shin (Group C)
Date of last modification: April 17, 2024
"""
import math
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt, colors
from matplotlib.animation import FuncAnimation
import random

rand = random.Random()

TIME_INTERVAL = 1  # Number of seconds the time step represents.

ON_RAMP_DESTINATION_A_PROP = 0.25  # Proportion of cars entering from the on ramp who need to change lanes.
UPSTREAM_CHANGE_LANE_PROP = 0.10  # Proportion of upstream traffic who still need to change lanes.

TOP_SPEED_MEAN = 100 / 3.6  # Mean top speed of a car (in m/s).
TOP_SPEED_SD = 6  # Standard deviation of top speed of a car.
TOP_SPEED_MIN = 80 / 3.6  # Minimum top speed of a car (in m/s).
TOP_SPEED_MAX = 120 / 3.6  # Maximum top speed of a car (in m/s).

ACCELERATION_MEAN = 6 / 3.6  # Mean acceleration of a car (in m/s^2).
ACCELERATION_SD = 1  # Standard deviation of acceleration of a car.

SPACE_MEAN = 2  # Mean following distance (in s).
SPACE_SD = 1  # Standard deviation of following distance of a car (in s).
SPACE_MAX = 10  # Maximum following distance of a car (in s).

INTERARRIVAL_MEAN = 1  # Mean interarrival time (in s).

CHANGE_TO_FAST_LANE_PROP = 0.10  # Proportion of cars that will change to the faster lane.

AREA_WIDTH = 15  # The y-dimension of the area.
AREA_LENGTH = 130  # The x-dimension of the area.
SIZE_OF_PIXEL = 5  # How many meters long/wide does one pixel represent.

Y_COORD_LEFT = 11  # The y-coordinate of the left lane.
Y_COORD_RIGHT = 8  # The y-coordinate of the right lane.
Y_COORD_ONRAMP = 5  # The y-coordinate of the on ramp.
X_COORD_ACCEL_START = 120  # The x-coordinate that the acceleration lane starts at.
X_COORD_ACCEL_END = 70  # The x-coordinate that the acceleration lane ends at.
X_COORD_START_LATE_MERGE = 80  # The x-coordinate that the late mergers start looking to merge at.
X_COORD_DESTINATION_START = 10  # The x-coordinate that the destination lanes start at.

# Setup for graphical display
colourmap = colors.ListedColormap(["lightgrey", "blue", "red", "yellow", "grey"])
normalizer = colors.Normalize(vmin=0.0, vmax=4.0)


class Car:
    """
    An agent representing a single car driving across a stretch of highway. The car's properties are defined in the
    __init__ function and their behaviour is specified in the step function.
    """

    def __init__(self, id, area, currentLane, earlyMerger, onRamp):
        """
        Initialize the car's properties including: id, the area object that they are driving in, their current
        lane, if they are an early merger, if they are entering from the on ramp, their destination lane, their top
        speed, their current speed, their acceleration, their desired following distance, and various variables
        to keep track of statistics.

        :param id: the car's id.
        :param area: the area object that the car is driving in.
        :param currentLane: the car's current lane, either "left" or "right".
        :param earlyMerger: True if the car is an early merger; False if the car is a late merger.
        :param onRamp: True if the car is entering from the on ramp; False if the car is entering from upstream traffic.
        """
        # Set attributes passed.
        self.id = id
        self.area = area
        self.currentLane = currentLane
        self.earlyMerger = earlyMerger
        self.onRamp = onRamp

        # Determine destination.
        if self.onRamp:
            if rand.random() < ON_RAMP_DESTINATION_A_PROP:
                self.destination = "a"
            else:
                self.destination = "b"
        else:
            if rand.random() < UPSTREAM_CHANGE_LANE_PROP:
                if self.currentLane == "left":
                    self.destination = "b"
                else:
                    self.destination = "a"
            else:
                if self.currentLane == "left":
                    self.destination = "a"
                else:
                    self.destination = "b"

        # Determine top speed.
        speed = np.random.normal(TOP_SPEED_MEAN, TOP_SPEED_SD)
        while speed < TOP_SPEED_MIN or speed > TOP_SPEED_MAX:
            speed = np.random.normal(TOP_SPEED_MEAN, TOP_SPEED_SD)
        self.topSpeed = speed

        # Set current speed.
        self.speed = speed

        # Determine acceleration.
        accel = np.random.normal(ACCELERATION_MEAN, ACCELERATION_SD)
        while accel <= 0:
            accel = np.random.normal(ACCELERATION_MEAN, ACCELERATION_SD)
        self.acceleration = accel

        # Determine desired following distance.
        space = np.random.lognormal(SPACE_MEAN, SPACE_SD)
        while space > SPACE_MAX:
            space = np.random.lognormal(SPACE_MEAN, SPACE_SD)
        self.space = space

        # Variables to keep track of stats.
        self.predictedWaitTime = self.topSpeed / (X_COORD_ACCEL_START - X_COORD_DESTINATION_START)
        self.enteredArea = False
        self.enterAreaTime = 0
        self.leftArea = False
        self.collectedWaitTime = False

    def step(self, time_step):
        """
        Defines the car's behaviour. Different behaviour is defined for cars in the lanes that connect to the upstream
        traffic, for cars that are early mergers on the on ramp, and for cars that are late mergers on the on ramp.
        For cars in the lanes that connect to the upstream traffic: cars merge into the lane that leads them to their
        destination lane, change lanes if the other lane is moving significantly faster (at least 10 km/h faster), then
        move forward. For cars that are early mergers on the on ramp: start looking for a gap to merge into as soon as
        they reach the acceleration ramp and then move forward, but if they didn't merge and they reach the end of the
        acceleration ramp then they come to a stop. For cars that are late mergers on the on ramp: start looking for
        a gap to merge into as soon as they reach the X_COORD_START_LATE_MERGE and then move forward, but if they
        didn't merge and they reach the end of the acceleration ramp then they come to a stop.

        :param time_step: the current time step.
        """
        # Behaviour of upstream cars.
        if not self.onRamp:
            # Either merge into the correct destination lane if required or change lanes if the other
            # lane is moving significantly faster.

            # Merge into the correct destination lane if required.
            if self.x <= X_COORD_ACCEL_START and not self.area.compare_lane_to_destination(self):
                if self.currentLane == "left" and not self.area.isoccupied(self.x, Y_COORD_RIGHT):
                    self.area.attemptmove(self, self.x, Y_COORD_RIGHT)
                    self.currentLane = "right"
                elif self.currentLane == "right" and not self.area.isoccupied(self.x, Y_COORD_LEFT):
                    self.area.attemptmove(self, self.x, Y_COORD_LEFT)
                    self.currentLane = "left"

            # Change lanes if other lane is moving significantly faster.
            elif self.x <= X_COORD_ACCEL_START and rand.random() < CHANGE_TO_FAST_LANE_PROP:
                # Get average speed of left and right lanes.
                speed_left, speed_right = self.area.get_lane_speeds()
                # If self is in left lane and the right lane is moving at least 10 km/hr faster, move
                # to the right lane.
                if self.currentLane == "left" and speed_right >= speed_left + (10 / 3.6) and \
                        not self.area.isoccupied(self.x, Y_COORD_RIGHT):
                    self.area.attemptmove(self, self.x, Y_COORD_RIGHT)
                    self.currentLane = "right"
                # If self is in the right lane and the left lane is moving at least 10 km/hr faster, move
                # to the left lane.
                elif self.currentLane == "right" and speed_left >= speed_right + (10 / 3.6) and \
                        not self.area.isoccupied(self.x, Y_COORD_LEFT):
                    self.area.attemptmove(self, self.x, Y_COORD_LEFT)
                    self.currentLane = "left"

            # Get the next x position.
            next_x = self.area.get_next_x(self)

            # Get the time they enter the area of interest.
            if not self.enteredArea and next_x <= X_COORD_ACCEL_START:
                self.enterAreaTime = time_step
                self.enteredArea = True

            # The car left the area of interest.
            if not self.leftArea and next_x <= X_COORD_DESTINATION_START:
                self.leftArea = True

            # If next x is out of the area, leave area. Otherwise, move to next x.
            if next_x <= 0 and self.area.compare_lane_to_destination(self):
                self.x = next_x
                self.area.leave_area(self)
            elif next_x <= 0 and not self.area.compare_lane_to_destination(self):
                print("Move rejected: attempting to leave at the wrong destination lane")
            else:
                self.area.attemptmove(self, next_x, self.y)

        # Behaviour of on ramp, early merging cars.
        elif self.onRamp and self.earlyMerger:
            # Start trying to merge as soon as they reach the acceleration ramp.
            if self.x <= X_COORD_ACCEL_START and not self.area.isoccupied(self.x, Y_COORD_RIGHT):
                self.area.attemptmove(self, self.x, Y_COORD_RIGHT)
                self.onRamp = False
                self.currentLane = "right"

            # Get the next x position.
            next_x = self.area.get_next_x(self)

            # Get the time they enter the area of interest.
            if not self.enteredArea and next_x <= X_COORD_ACCEL_START:
                self.enterAreaTime = time_step
                self.enteredArea = True

            # If next x is at or beyond the end of the acceleration ramp.
            if next_x <= X_COORD_ACCEL_END:
                # Find the car in front.
                car_in_front = self.area.get_car_in_front(self.id, self.y)
                # If there is a car in front, move to one spot behind them.
                if car_in_front is not None:
                    next_x = car_in_front.x + 1
                    self.area.attemptmove(self, next_x, self.y)
                # Otherwise, move to the end of the acceleration ramp.
                else:
                    next_x = X_COORD_ACCEL_END
                    self.area.attemptmove(self, next_x, self.y)
            else:
                self.area.attemptmove(self, next_x, self.y)

        # Behaviour of on ramp, late merging cars.
        elif self.onRamp and not self.earlyMerger:
            # Start trying to merge as soon as they reach the X_COORD_START_LATE_MERGE.
            if self.x <= X_COORD_START_LATE_MERGE and not self.area.isoccupied(self.x, Y_COORD_RIGHT):
                self.area.attemptmove(self, self.x, Y_COORD_RIGHT)
                self.onRamp = False
                self.currentLane = "right"

            # Get the next x position.
            next_x = self.area.get_next_x(self)

            # Get the time they enter the area of interest.
            if not self.enteredArea and next_x <= X_COORD_ACCEL_START:
                self.enterAreaTime = time_step
                self.enteredArea = True

            # If next x is at or beyond the end of the acceleration ramp.
            if next_x <= X_COORD_ACCEL_END:
                # Find the car in front.
                car_in_front = self.area.get_car_in_front(self.id, self.y)
                # If there is a car in front, move to one spot behind them.
                if car_in_front is not None:
                    next_x = car_in_front.x + 1
                    self.area.attemptmove(self, next_x, self.y)
                # Otherwise, move to the end of the acceleration ramp.
                else:
                    next_x = X_COORD_ACCEL_END
                    self.area.attemptmove(self, next_x, self.y)
            else:
                self.area.attemptmove(self, next_x, self.y)

    def __str__(self):
        """
        Define the string representation of Car. It includes the car's id, x position, and y position.

        :return: the string representation of a car.
        """
        return "id: %d  x: %d  y: %d" % (self.id, self.x, self.y)


class Area:
    """
    The area that the agent drives through. It controls the agents' positions and movement.

    This class was originally written by Reid Kerr for use in an assignment in BDA450 but has been modified by Group C.
    """

    def __init__(self):
        """
        Initialize the area's properties including: the storage data structure, the id to be used for the next agent to
        enter the area, the next arrival times, and various variables to keep track of statistics.
        """
        # Tracking of positions of agents
        self.storage = AreaGrid()

        # Tracking of next car id.
        self.next_id = 1

        # Tracking of next arrival times.
        self.next_left_arrival_time = math.ceil(rand.expovariate(1 / INTERARRIVAL_MEAN))
        self.next_right_arrival_time = math.ceil(rand.expovariate(1 / INTERARRIVAL_MEAN))
        self.next_onramp_arrival_time = math.ceil(rand.expovariate(1 / INTERARRIVAL_MEAN))

        # Variables to track statistics.
        self.avgSpeedEachTimestep = []
        self.chgSpeedTimeSteps = []
        self.avgWaitTimeEachTimestep = []
        self.chgWaitTimeTimeStep = []
        self.numCarsLeft = 0

        # Bitmap is for graphical display
        self.bitmap = [[0.0 for i in range(AREA_LENGTH)] for j in range(AREA_WIDTH)]

    def enter_area(self, car, x, y):
        """
        An agent must enter the area in an unoccupied space at the right end of the area. The function returns True
        if the agent is successfully added to the area; False otherwise.

        This function was originally written by Reid Kerr for use in an assignment in BDA450 but has been modified by
        Group C.

        :param car: the car to put into the area.
        :param x: the x position of the car
        :param y: the y position of the car
        :return: True if the agent is successfully added to the area; False otherwise.
        """
        # New entrant to the area, must attempt to start at one end
        if x != AREA_LENGTH - 1:
            print("Must start at an end!")
            return False
        # Only allow move if space not currently occupied
        if self.storage.isoccupied(x, y):
            print("Move rejected: occupied")
            return False
        self.storage.add_item(x, y, car)
        car.x = x
        car.y = y
        return True

    def leave_area(self, car):
        """
        An agent must leave the area at the left end of the area in the correct destination lane. The function returns
        True if the agent is successfully removed from the area; False otherwise.

        This function was originally written by Reid Kerr for use in an assignment in BDA450 but has been modified by
        Group C.

        :param car: the car to leave the area.
        :return: True if the agent is successfully removed from the area; False otherwise.
        """
        # Must attempt to leave at one end
        if car.x > 0:
            print("Must leave at an end!")
            return False
        # Must attempt to leave at correct destination.
        if not self.compare_lane_to_destination(car):
            print("Must leave at correct destination!", car.id)
            return False
        self.storage.remove_item(car)

    def attemptmove(self, car, x, y):
        """
        Moves an agent in the area. The agent can only move forward (i.e. towards the left end) within the area and
        only to unoccupied spaces. The function returns True if the agent is successfully moved in the area; False
        otherwise.

        This function was originally written by Reid Kerr for use in an assignment in BDA450 but has been modified by
        Group C.

        :param car: the car to be moved.
        :param x: the x position to move the car to.
        :param y: the y position to move the car to.
        :return: True if the agent is successfully moved in the area; False otherwise.
        """
        # Only allows moves within area
        if x < 0.0 or x >= AREA_LENGTH or y < 0 or y >= AREA_WIDTH:
            print("Move rejected: out of area!", car.x, x)
            return False
        # Only allows cars to move forward.
        if car.x - x < 0:
            print("Move rejected: car cannot move backwards!")
            return False
        # Only allow move if space not currently occupied
        if self.storage.isoccupied(x, y):
            return False
        car.x = x
        car.y = y
        self.storage.move_item(x, y, car)
        return True

    def isoccupied(self, x, y):
        """
        Returns true if x,y is occupied by an agent; false otherwise.

        This function was originally written by Reid Kerr for use in an assignment in BDA450.

        :param x: the x-position you want to check for occupation.
        :param y: the y-position you want to check for occupation.
        :return: True if x,y is occupied by an agent; false otherwise.
        """
        return self.storage.isoccupied(x, y)

    def refresh_image(self):
        """
        Updates the graphic for display.

        This function was originally written by Reid Kerr for use in an assignment in BDA450 but has been modified by
        Group C.
        """
        self.bitmap = [[0.0 for i in range(AREA_LENGTH)] for j in range(AREA_WIDTH)]
        for car in self.storage.get_list():
            x = round(car.x)
            y = round(car.y)
            colour = 1
            self.bitmap[y][x] = colour

    def sort_by_x(self):
        """
        Returns a list of all the agents in the area sorted by their x-position. Agents that are closer to the left end
        of the area come before agents that are closer to the right end of the area.

        :return: a list of all agents in the area sorted by their x-position.
        """
        if len(self.storage.get_list()) > 0:
            sorted_list = []
            for car in self.storage.get_list():
                sorted_list.append(car)
            sorted_list.sort(key=lambda car: abs(car.x))
            return sorted_list

    def get_car_in_front(self, id, y):
        """
        Returns the agent directly in front of the car with the given id. Returns None if there is no agent in front.

        :param id: the id of the agent that you want to find the agent in front of.
        :param y: the y position of the agent that you want to find the agent in front of.
        :return: the agent in front; None if there is no agent in front of.
        """
        # Get list of all cars sorted by x.
        all_cars = self.sort_by_x()
        # Gat all cars in same lane.
        if len(all_cars) > 1:
            same_lane = []
            for car in all_cars:
                if car.y == y:
                    same_lane.append(car)
            # Find index of our car and return car in front of us.
            for i in range(len(same_lane)):
                if same_lane[i].id == id and i > 0:
                    return same_lane[i - 1]
        return None

    def get_next_x(self, car):
        """
        Returns the next x-position car should move to based on their speed and the distance they wish to maintain
        between them and the car in front of them.

        :param car: the car you want to determine the next x-position of.
        :return: the next x position car should move to.
        """
        # Find distance between self and car in front
        car_in_front = self.get_car_in_front(car.id, car.y)
        # If there is a car in front.
        if car_in_front is not None:
            distance = abs(car.x - car_in_front.x) * SIZE_OF_PIXEL
            # Find speed.
            car.speed = min(distance / car.space, car.topSpeed)
            # Find change in x-position.
            displacement = car.speed * TIME_INTERVAL + 0.5 * car.acceleration * TIME_INTERVAL * TIME_INTERVAL
        else:
            # Find speed.
            car.speed = min(car.speed + car.acceleration * TIME_INTERVAL, car.topSpeed)
            # Find change in x-position.
            displacement = car.speed * TIME_INTERVAL + 0.5 * car.acceleration * TIME_INTERVAL * TIME_INTERVAL
        # Find next x position.
        next_x = car.x - (displacement / SIZE_OF_PIXEL)
        return next_x

    def compare_lane_to_destination(self, car):
        """
        Returns True if the car is in the lane that leads towards their desired destination; False otherwise. The left
        lane leads to destination 'a' and the right lane leads to destination 'b'.

        :param car: the car whose current lane and destination you want to compare.
        :return: True if the car is in the lane that leads towards their desired destination.
        """
        if car.currentLane == 'left' and car.destination == 'a':
            return True
        elif car.currentLane == 'right' and car.destination == 'b':
            return True
        else:
            return False

    def get_lane_speeds(self):
        """
        Returns a tuple, where the first element represents the average speed of all the cars in the left lane and the
        second element represents the average speed of all the cars in the right lane. Returns 0 if there are no cars
        in that lane.

        :return: a tuple, where the first element represents the average speed of all the cars in the left lane and the
        second element represents the average speed of all the cars in the right lane.
        """
        left_lane_speeds = []
        right_lane_speeds = []
        # Get a list of all cars in the left lane and a list of all cars in the right lane.
        for car in self.storage.get_list():
            if car.currentLane == 'left':
                left_lane_speeds.append(car.speed)
            elif car.currentLane == 'right':
                right_lane_speeds.append(car.speed)
        # There is no cars in the left lane, but at least one in the right.
        if len(left_lane_speeds) == 0 and len(right_lane_speeds) != 0:
            return (0, sum(right_lane_speeds) / len(right_lane_speeds))
        # There is at least one car in the left lane, but none in the right.
        elif len(left_lane_speeds) != 0 and len(right_lane_speeds) == 0:
            return (sum(left_lane_speeds) / len(left_lane_speeds), 0)
        # There are cars in both lanes.
        elif len(left_lane_speeds) != 0 and len(right_lane_speeds) != 0:
            return (sum(left_lane_speeds) / len(left_lane_speeds), sum(right_lane_speeds) / len(right_lane_speeds))
        # There are no cars in either lane.
        else:
            return (0, 0)

    def run_step(self, time_step, earlyMerger):
        """
        This function is called at each time step. A new car is added to the left, right, and/or on ramp if the
        pre-determined time has passed. Then, the behaviour of each car is performed in order of x-position, such that
        cars in front move before the cars behind them. Statistics are tracked and the graphics are refreshed.

        :param time_step: the current timestep.
        :param earlyMerger: True if the cars in this simulation are early mergers; False if they are late mergers.
        """
        # Time for a new car to arrive in the left upstream lane.
        if time_step == self.next_left_arrival_time:
            # Create a new car.
            car = Car(self.next_id, self, "left", earlyMerger, False)
            # Add car to area.
            self.enter_area(car, AREA_LENGTH - 1, 11)
            # Change the next arrival time for the left upstream lane.
            amnt_incr = math.ceil(rand.expovariate(1 / INTERARRIVAL_MEAN))
            self.next_left_arrival_time += amnt_incr
            self.next_id += 1

        # Time for a new car to arrive in the right upstream lane.
        if time_step == self.next_right_arrival_time:
            # Create a new car.
            car = Car(self.next_id, self, "right", earlyMerger, False)
            # Add car to area.
            self.enter_area(car, AREA_LENGTH - 1, 8)
            # Change the next arrival time for the right upstream lane.
            amnt_incr = math.ceil(rand.expovariate(1 / INTERARRIVAL_MEAN))
            self.next_right_arrival_time += amnt_incr
            self.next_id += 1

        # Time for a new car to arrive in the on ramp.
        if time_step == self.next_onramp_arrival_time:
            # Create new car.
            car = Car(self.next_id, self, "onramp", earlyMerger, True)
            # Add car to area.
            self.enter_area(car, AREA_LENGTH - 1, 5)
            # Change the next arrival time for the on ramp.
            amnt_incr = math.ceil(rand.expovariate(1 / INTERARRIVAL_MEAN))
            self.next_onramp_arrival_time += amnt_incr
            self.next_id += 1

        # Track the speed of all cars and waiting time of all cars.
        speeds = []
        waitTimes = []

        # Call behaviour of each car, in order of x-position.
        sorted_cars = self.sort_by_x()
        if sorted_cars is not None:
            for car in sorted_cars:
                car.step(time_step)
                # If car is in the area of interest, collect their speed.
                if car.enteredArea and not car.leftArea:
                    speeds.append(car.speed)
                # If car has left the area of interest in this timestep,
                # increment self.numCarsLeft and collect the car's waiting time.
                if car.leftArea and not car.collectedWaitTime:
                    self.numCarsLeft += 1
                    actual_wait_time = (time_step - car.enterAreaTime) * TIME_INTERVAL
                    waitTimes.append(actual_wait_time - car.predictedWaitTime)
                    car.collectedWaitTime = True

        # Calculate the average speed and wait time at this time step.
        if len(speeds) != 0:
            self.avgSpeedEachTimestep.append(np.mean(speeds))
            self.chgSpeedTimeSteps.append(time_step)
        if len(waitTimes) != 0:
            self.avgWaitTimeEachTimestep.append(np.mean(waitTimes))
            self.chgWaitTimeTimeStep.append(time_step)

        # Refresh graphics
        self.refresh_image()


class AreaGrid:
    """
    This class is used to provide storage, lookup of occupants of area.

    This class and all functions in it were originally written by Reid Kerr for use in an assignment in BDA450.
    """

    def __init__(self):
        """
        Initialize the data structure to be used as storage.
        """
        self.dic = dict()

    def isoccupied(self, x, y):
        """
        Returns true if x,y is in the storage; false otherwise.

        :param x: the x-position to check.
        :param y: the y-position to check.
        :return: True if x,y is in the storage; false otherwise.
        """
        return (x, y) in self.dic

    def add_item(self, x, y, item):
        """
        Stores item at coordinates x, y.  Throws an exception if the coordinates are invalid.  Returns false if
        unsuccessful (e.g., the square is occupied) or true if successful.

        :param x: the x-position to store the item at.
        :param y: the y-position to store the item at.
        :param item: the item to store.
        :return: True if successful; False otherwise.
        """
        self.check_coordinates(x, y)
        if (x, y) in self.dic:
            return False
        self.dic[(x, y)] = item
        return True

    def move_item(self, x, y, item):
        """
        Removes item from its current coordinates (which do not need to be provided) and stores it
        at coordinates x, y.  Throws an exception if the coordinates are invalid or if the square is occupied.

        :param x: the new x-position to store.
        :param y: the new y-position to store.
        :param item: the item whose coordinates are to be changed.
        """
        self.check_coordinates(x, y)
        if self.isoccupied(x, y):
            raise Exception("Move to occupied square!")

        # Find and remove previous location.  Assumed state is valid (meaning only one entry per x,y key)
        oldloc = next(key for key, value in self.dic.items() if value == item)
        del self.dic[oldloc]
        self.add_item(x, y, item)

    def remove_item(self, item):
        """
        Removes item (coordinates do not need to be provided). Throws an exception if the item doesn't exist.

        :param item: the item to be removed.
        """
        # Find and remove previous location.  Assumed state is valid (meaning only one entry per x,y key)
        oldloc = next(key for key, value in self.dic.items() if value == item)
        if oldloc is None:
            raise Exception('Attempt to remove non-existent item!')
        del self.dic[oldloc]

    def get_item(self, x, y):
        """
        Returns the item with the provided x, y coordinates.

        :param x: the x-coordinate of the item to be returned.
        :param y: the y-coordinate of the item to be returned.
        :return: the item with the provided x, y coordinates.
        """
        return self.dic.get((x, y), None)

    def get_list(self):
        """
        Returns a list of all agents in the simulation.

        :return: a list of all agents in the simulation.
        """
        return list(self.dic.values())

    def check_coordinates(self, x, y):
        """
        Checks if there is an item with the provided x, y coordinates. Throws an exception if the x, y coordinates
        provided are outside the area.

        :param x: the x coordinate to check.
        :param y: the y coordinate to check.
        """
        if x < 0 or x >= AREA_LENGTH or y < 0 or y >= AREA_WIDTH:
            raise Exception("Illegal coordinates!")


class MergingSimulation:
    """
    This class runs the car merging simulation.
    """

    def __init__(self):
        """
        Initialize the simulation's attributes, including the current timestep and statistics.
        """
        # Track timestep
        self.t = 0

        # Track statistics.
        self.avgSpeeds = None
        self.avgSpeedTimeSteps = None
        self.avgWait = None
        self.avgWaitTimeSteps = None
        self.throughput = None

    def run_sim(self, graphics, earlyMerger, numTimeSteps):
        """
        Runs the simulation.

        :param graphics: True if the user wants graphics to be shown; False otherwise.
        :param earlyMerger: True if the user wants all cars in the simulation to be early mergers; False if the user
        wants all cars in the simulation to be late mergers.
        :param numTimeSteps: the total number of time steps to run the simulation for.
        """
        # Initialize the area.
        sw = Area()

        # The user wants graphics to be shown.
        if graphics:
            fig, ax = plt.subplots(figsize=(15, 8))

            # Draw lane lines
            for y in [7, 10]:
                ax.plot([0, AREA_LENGTH], [y, y], color='gray', linestyle='--')
            image = ax.imshow(sw.bitmap, cmap=colourmap, norm=normalizer, animated=True)

            def updatefigure(*args):
                self.t += 1
                if self.t % 100 == 0:
                    print("Time: %d" % self.t)
                sw.run_step(self.t, earlyMerger)
                sw.refresh_image()
                image.set_array(sw.bitmap)
                return image,

            anim = FuncAnimation(fig, updatefigure, frames=numTimeSteps, interval=1000, blit=False, repeat=False)
            plt.show()

        # The user does not want graphics to be shown.
        else:
            while self.t <= numTimeSteps:
                if self.t % 100 == 0:
                    print("Time: %d" % self.t)
                sw.run_step(self.t, earlyMerger)
                self.t += 1

        print("Done!")

        # Update statistic attributes.
        self.avgSpeeds = sw.avgSpeedEachTimestep
        self.avgSpeedTimeSteps = sw.chgSpeedTimeSteps
        self.avgWait = sw.avgWaitTimeEachTimestep
        self.avgWaitTimeSteps = sw.chgWaitTimeTimeStep
        self.throughput = sw.numCarsLeft / (numTimeSteps * TIME_INTERVAL / 3600)


if __name__ == '__main__':
    '''
    # Determining the time step the steady state begins for speed for early mergers.
    sim = MergingSimulation()
    sim.run_sim(False, True, 1400)
    plt.figure(1)
    plt.plot(sim.avgSpeedTimeSteps, sim.avgSpeeds)
    plt.xlabel("Time step")
    plt.ylabel("Average speed of cars (m/s)")
    plt.title("Average speed of early merging cars each time step")
    plt.show()

    # Determining the time step the steady state begins for wait time for early mergers.
    sim = MergingSimulation()
    sim.run_sim(False, True, 1400)
    plt.figure(2)
    plt.plot(sim.avgWaitTimeSteps, sim.avgWait)
    plt.xlabel("Time step")
    plt.ylabel("Average wait time of cars (s)")
    plt.title("Average wait time of early merging cars each time step")
    plt.show()

    # Determining the time step the steady state begins for speed for late mergers.
    sim = MergingSimulation()
    sim.run_sim(False, False, 1400)
    plt.figure(3)
    plt.plot(sim.avgSpeedTimeSteps, sim.avgSpeeds)
    plt.xlabel("Time step")
    plt.ylabel("Average speed of cars (m/s)")
    plt.title("Average speed of late merging cars each time step")
    plt.show()

    # Determining the time step the steady state begins for wait time for late mergers.
    sim = MergingSimulation()
    sim.run_sim(False, False, 1400)
    plt.figure(4)
    plt.plot(sim.avgWaitTimeSteps, sim.avgWait)
    plt.xlabel("Time step")
    plt.ylabel("Average wait time of cars (s)")
    plt.title("Average wait time of late merging cars each time step")
    plt.show() 
    '''
    # Running the simulation for early mergers.
    early_avg_speeds = []
    early_avg_wait = []
    early_throughput = []
    early_all_speeds = []
    early_all_wait = []
    for i in range(5):
        print("\nRunning simulation %d for early mergers" % (i + 1))
        sim = MergingSimulation()
        sim.run_sim(False, True, 1400)
        # Find indexes of the beginning of the steady state.
        for i in range(len(sim.avgSpeedTimeSteps)):
            if sim.avgSpeedTimeSteps[i] >= 400:
                beg_steady_state_speed_index = i
                break
        for i in range(len(sim.avgWaitTimeSteps)):
            if sim.avgWaitTimeSteps[i] >= 600:
                beg_steady_state_wait_index = i
                break
        # Discard the warm up.
        steady_state_speed = sim.avgSpeeds[beg_steady_state_speed_index:]
        steady_state_wait = sim.avgWait[beg_steady_state_wait_index:]
        # Keep speeds and wait times.
        early_all_speeds = early_all_speeds + steady_state_speed
        early_all_wait = early_all_wait + steady_state_wait
        # Calculate average speeds and wait time for steady state.
        early_avg_speeds.append(np.mean(steady_state_speed))
        early_avg_wait.append(np.mean(steady_state_wait))
        early_throughput.append(sim.throughput)

    # Running the simulation for late mergers.
    late_avg_speeds = []
    late_avg_wait = []
    late_throughput = []
    late_all_speeds = []
    late_all_wait = []
    for i in range(5):
        print("\nRunning simulation %d for late mergers" % (i + 1))
        sim = MergingSimulation()
        sim.run_sim(False, False, 1400)
        # Find indexes of the beginning of the steady state.
        for i in range(len(sim.avgSpeedTimeSteps)):
            if sim.avgSpeedTimeSteps[i] >= 400:
                beg_steady_state_speed_index = i
                break
        for i in range(len(sim.avgWaitTimeSteps)):
            if sim.avgWaitTimeSteps[i] >= 600:
                beg_steady_state_wait_index = i
                break
        # Discard the warm up.
        steady_state_speed = sim.avgSpeeds[beg_steady_state_speed_index:]
        steady_state_wait = sim.avgWait[beg_steady_state_wait_index:]
        # Keep speeds and wait times.
        late_all_speeds = late_all_speeds + steady_state_speed
        late_all_wait = late_all_wait + steady_state_wait
        # Calculate average speeds and wait time for steady state.
        late_avg_speeds.append(np.mean(steady_state_speed))
        late_avg_wait.append(np.mean(steady_state_wait))
        late_throughput.append(sim.throughput)

    # Calculate statistics for early mergers.
    print("\nStatistics for early mergers during steady state:")
    print("For average speed (i.e. speed of traffic flow) (m/s):")
    print("Mean: %0.4f" % np.mean(early_avg_speeds))
    print("Standard deviation: %0.4f" % np.std(early_avg_speeds))
    lower_limit, upper_limit = st.t.interval(confidence=0.95,
                                             df=len(early_avg_speeds) - 1,
                                             loc=np.mean(early_avg_speeds),
                                             scale=st.sem(early_avg_speeds))
    print("95%% confidence interval limits: %0.4f, %0.4f" % (lower_limit, upper_limit))
    plt.figure(5)
    plt.title("Histogram of speed of traffic flow during steady state for early mergers")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency")
    plt.hist(early_all_speeds)

    print("\nFor waiting times (s):")
    print("Mean: %0.4f" % np.mean(early_avg_wait))
    print("Standard deviation: %0.4f" % np.std(early_avg_wait))
    lower_limit, upper_limit = st.t.interval(confidence=0.95,
                                             df=len(early_avg_wait) - 1,
                                             loc=np.mean(early_avg_wait),
                                             scale=st.sem(early_avg_wait))
    print("95%% confidence interval limits: %0.4f, %0.4f" % (lower_limit, upper_limit))
    plt.figure(6)
    plt.title("Histogram of waiting times during steady state for early mergers")
    plt.xlabel("Waiting times (s)")
    plt.ylabel("Frequency")
    plt.hist(early_all_wait)

    print("\nFor throughput (cars/hr):")
    print("Mean: %0.4f" % np.mean(early_throughput))
    print("Standard deviation: %0.4f" % np.std(early_throughput))
    lower_limit, upper_limit = st.t.interval(confidence=0.95,
                                             df=len(early_throughput) - 1,
                                             loc=np.mean(early_throughput),
                                             scale=st.sem(early_throughput))
    print("95%% confidence interval limits: %0.4f, %0.4f" % (lower_limit, upper_limit))

    # Calculate statistics for late mergers.
    print("\nStatistics for late mergers during steady state:")
    print("For average speed (i.e. speed of traffic flow) (m/s):")
    print("Mean: %0.4f" % np.mean(late_avg_speeds))
    print("Standard deviation: %0.4f" % np.std(late_avg_speeds))
    lower_limit, upper_limit = st.t.interval(confidence=0.95,
                                             df=len(late_avg_speeds) - 1,
                                             loc=np.mean(late_avg_speeds),
                                             scale=st.sem(late_avg_speeds))
    print("95%% confidence interval limits: %0.4f, %0.4f" % (lower_limit, upper_limit))
    plt.figure(7)
    plt.title("Histogram of speed of traffic flow during steady state for late mergers")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency")
    plt.hist(late_all_speeds)

    print("\nFor waiting times (s):")
    print("Mean: %0.4f" % np.mean(late_avg_wait))
    print("Standard deviation: %0.4f" % np.std(late_avg_wait))
    lower_limit, upper_limit = st.t.interval(confidence=0.95,
                                             df=len(late_avg_wait) - 1,
                                             loc=np.mean(late_avg_wait),
                                             scale=st.sem(late_avg_wait))
    print("95%% confidence interval limits: %0.4f, %0.4f" % (lower_limit, upper_limit))
    plt.figure(8)
    plt.title("Histogram of waiting times during steady state for late mergers")
    plt.xlabel("Waiting times (s)")
    plt.ylabel("Frequency")
    plt.hist(late_all_wait)

    print("\nFor throughput (cars/hr):")
    print("Mean: %0.4f" % np.mean(late_throughput))
    print("Standard deviation: %0.4f" % np.std(late_throughput))
    lower_limit, upper_limit = st.t.interval(confidence=0.95,
                                             df=len(late_throughput) - 1,
                                             loc=np.mean(late_throughput),
                                             scale=st.sem(late_throughput))
    print("95%% confidence interval limits: %0.4f, %0.4f" % (lower_limit, upper_limit))

    plt.show()



