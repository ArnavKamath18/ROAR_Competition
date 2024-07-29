import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from typing import List, Type, Optional, Dict, Any
import carla
import numpy as np
import asyncio

class RoarCompetitionRule:
    def __init__(self, waypoints: List[roar_py_interface.RoarPyWaypoint], vehicle: roar_py_carla.RoarPyCarlaActor, world: roar_py_carla.RoarPyCarlaWorld) -> None:
        self.waypoints = waypoints
        self.vehicle = vehicle
        self.world = world
        self._last_vehicle_location = vehicle.get_3d_location()
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = vehicle.get_roll_pitch_yaw().copy()
        self.furthest_waypoints_index = 0

    def initialize_race(self):
        self._last_vehicle_location = self.vehicle.get_3d_location()
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()
        closest_waypoint_idx = self._find_closest_waypoint_index(self._last_vehicle_location)
        self.waypoints = self.waypoints[closest_waypoint_idx+1:] + self.waypoints[:closest_waypoint_idx+1]
        print(f"Total waypoints: {len(self.waypoints)}")

    def _find_closest_waypoint_index(self, location: np.ndarray) -> int:
        return min(range(len(self.waypoints)), key=lambda i: np.linalg.norm(location - self.waypoints[i].location))

    def lap_finished(self, check_step=5) -> bool:
        return self.furthest_waypoints_index + check_step >= len(self.waypoints)

    async def tick(self, check_step=15):
        current_location = self.vehicle.get_3d_location()
        delta_vector = current_location - self._last_vehicle_location
        delta_vector_norm = np.linalg.norm(delta_vector)
        delta_vector_unit = delta_vector / delta_vector_norm if delta_vector_norm >= 1e-5 else np.zeros(3)
        
        min_dis, min_index = np.inf, 0
        for i, waypoint in enumerate(self.waypoints[self.furthest_waypoints_index:self.furthest_waypoints_index + check_step]):
            projection = np.dot(waypoint.location - current_location, delta_vector_unit)
            projection = np.clip(projection, 0, delta_vector_norm)
            distance = np.linalg.norm(waypoint.location - (current_location + projection * delta_vector_unit))
            if distance < min_dis:
                min_dis = distance
                min_index = i

        self.furthest_waypoints_index += min_index
        self._last_vehicle_location = current_location
        print(f"Reached waypoint {self.furthest_waypoints_index} at {self.waypoints[self.furthest_waypoints_index].location}")

    async def respawn(self):
        self.vehicle.set_transform(self._respawn_location, self._respawn_rpy)
        self.vehicle.set_linear_3d_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        await asyncio.gather(*(self.world.step() for _ in range(20)))
        self._last_vehicle_location = self.vehicle.get_3d_location()
        self.furthest_waypoints_index = 0

async def evaluate_solution(world: roar_py_carla.RoarPyCarlaWorld, solution_constructor: Type[RoarCompetitionSolution], max_seconds=12000, enable_visualization=False) -> Optional[Dict[str, Any]]:
    viewer = ManualControlViewer() if enable_visualization else None
    vehicle, sensors = await spawn_vehicle_and_sensors(world)
    solution = solution_constructor(
        world.maneuverable_waypoints,
        RoarCompetitionAgentWrapper(vehicle),
        sensors["camera"],
        sensors["location"],
        sensors["velocity"],
        sensors["rpy"],
        sensors["occupancy_map"],
        sensors["collision"]
    )

    rule = RoarCompetitionRule(world.maneuverable_waypoints * 3, vehicle, world)
    await asyncio.gather(*(world.step() for _ in range(20)))
    rule.initialize_race()

    await vehicle.receive_observation()
    await solution.initialize()
    start_time = world.last_tick_elapsed_seconds

    while True:
        if (world.last_tick_elapsed_seconds - start_time) > max_seconds:
            vehicle.close()
            return None

        await vehicle.receive_observation()
        await rule.tick()

        if np.linalg.norm(sensors["collision"].get_last_observation().impulse_normal) > 100.0:
            print("Major collision detected, respawning...")
            await rule.respawn()

        if rule.lap_finished():
            break

        if enable_visualization and viewer.render(sensors["camera"].get_last_observation()) is None:
            vehicle.close()
            return None

        await solution.step()
        await world.step()

    vehicle.close()
    if viewer:
        viewer.close()

    return {"elapsed_time": world.last_tick_elapsed_seconds - start_time}

async def spawn_vehicle_and_sensors(world: roar_py_carla.RoarPyCarlaWorld):
    waypoints = world.maneuverable_waypoints
    vehicle = world.spawn_vehicle("vehicle.tesla.model3", waypoints[0].location + np.array([0,0,1]), waypoints[0].roll_pitch_yaw, True)
    if not vehicle:
        raise RuntimeError("Failed to spawn vehicle")

    sensors = {
        "camera": vehicle.attach_camera_sensor(roar_py_interface.RoarPyCameraSensorDataRGB, np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), np.array([0, 10/180.0*np.pi, 0]), image_width=1024, image_height=768),
        "location": vehicle.attach_location_in_world_sensor(),
        "velocity": vehicle.attach_velocimeter_sensor(),
        "rpy": vehicle.attach_roll_pitch_yaw_sensor(),
        "occupancy_map": vehicle.attach_occupancy_map_sensor(50, 50, 2.0, 2.0),
        "collision": vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))
    }

    if not all(sensors.values()):
        raise RuntimeError("Failed to attach all necessary sensors")

    return vehicle, sensors

async def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)
    
    evaluation_result = await evaluate_solution(world, RoarCompetitionSolution, max_seconds=5000, enable_visualization=True)
    
    if evaluation_result:
        print(f"Solution finished in {evaluation_result['elapsed_time']} seconds")
    else:
        print("Solution failed to finish in time")

if __name__ == "__main__":
    asyncio.run(main())
