"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
from mpc import MPCController, State


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
        self.ref_line = np.array([[i,waypoint.location[0], waypoint.location[1], waypoint.roll_pitch_yaw[2], 0, 25, 0] for i,waypoint in enumerate(maneuverable_waypoints)])
        # self.ref_line = np.genfromtxt('traj_race_cl_mintime.csv', delimiter=';', skip_header=3)
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        self.MPC = MPCController(
            dt=0.05,
            horizon=10,
            reference_trajectory=self.ref_line
        )
        self.current_time = 0

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()

        self.MPC.prev_x, self.MPC.prev_y, self.MPC.prev_yaw = vehicle_location[0], vehicle_location[1], vehicle_rotation[2]
        print("Initial x, y, yaw:", self.MPC.prev_x, self.MPC.prev_y, self.MPC.prev_yaw)

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        self.current_time = 0


    async def step(
        self,
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        slip_angle = np.arctan2(vehicle_velocity[1], vehicle_velocity[0]) - vehicle_rotation[2]
        slip_angle = normalize_rad(slip_angle)
        state = State(x=vehicle_location[0],
                        y=vehicle_location[1],
                        steering_angle=0,
                        velocity=vehicle_velocity_norm,
                        yaw_angle=vehicle_rotation[2],
                        yaw_rate=0, #will be calculated in mpc
                        slip_angle=slip_angle)
        optimal_control = self.MPC.solve_mpc(state, current_time=self.current_time)
        assert optimal_control is not None
        assert len(optimal_control) == 2
        assert optimal_control[0] <= 1.0 and optimal_control[0] >= -1.0
        assert optimal_control[1] <= 1.0 and optimal_control[1] >= -1.0
        control = {
            "throttle": np.clip(optimal_control[1], 0.0, 1.0),
            "steer": np.clip(optimal_control[0], -1.0, 1.0),
            "brake": np.clip(-optimal_control[1], 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        self.current_time += 0.05
        return control