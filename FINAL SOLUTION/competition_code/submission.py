"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
from simple_pid import PID
from pure_pursuit import PurePursuit

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.ref_line = np.genfromtxt('C:/Users/shoun\Documents/UCB ROAR 2024/Competition Base (Virgile Foussereau 3)/ROAR_Competition-submission/competition_code/traj_race_cl_mintime.csv', delimiter=';', skip_header=3)
        self.pid_longitudinal = PID(0.39, 0.1, 0.0699, setpoint=0)
        self.pid_longitudinal.output_limits = (-1, 1)

        self.pure_pursuit = PurePursuit(np.array([waypoint.location for waypoint in self.maneuverable_waypoints]))

    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        self.pos_diff_speed = 0
        self.neg_diff_speed = 0
        self.count = 0
        self.max_speed_reached = 0
        self.speed_diff_at_max = 0


    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the point closest to the vehicle on the reference trajectory
        dists = np.linalg.norm(self.ref_line[:,1:3] - vehicle_location[:2], axis=1)
        closest_idx = np.argmin(dists)

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        
        L = max(0.5, 0.8 * vehicle_velocity_norm)
        goalpt, lastidx, steer_control = self.pure_pursuit.step(vehicle_location[:2], vehicle_rotation[2], L, self.current_waypoint_idx, vehicle_velocity_norm)
       
        self.pid_longitudinal.setpoint = self.ref_line[closest_idx][5] + 10
        
        if self.current_waypoint_idx <= 100:
            throttle_control = 1.0
        else:
            throttle_control = self.pid_longitudinal(vehicle_velocity_norm, dt=0.05)
            if throttle_control >= 0.8:
                throttle_control = 1.0
            elif throttle_control >= 0.5:
                throttle_control += 0.5 * ((throttle_control - 0.5) / 0.3)
            elif throttle_control <= -0.4:
                throttle_control += 0.3 * ((0.4 + throttle_control) / -0.6)

        print(self.current_waypoint_idx, throttle_control, steer_control, -throttle_control)

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": np.clip(steer_control, -1.0, 1.0),
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control