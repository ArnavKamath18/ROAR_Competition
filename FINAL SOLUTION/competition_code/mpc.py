import numpy as np
from scipy.optimize import minimize
from shapely.geometry import LineString, Point

from vehiclemodels.init_st import init_st
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from scipy.integrate import odeint
import bisect

def func_ST(x, t, u, p):
    f = vehicle_dynamics_st(x, u, p)
    return f

class State:
    def __init__(self, x=0.0, y=0.0, steering_angle=0.0, velocity=0.0, yaw_angle=0.0, yaw_rate=0.0, slip_angle=0.0):
        self.x = x
        self.y = y
        self.steering_angle = steering_angle
        self.velocity = velocity
        self.yaw_angle = yaw_angle
        self.yaw_rate = yaw_rate
        self.slip_angle = slip_angle

    def __str__(self):
        return f"State(x={self.x}, y={self.y}, steering_angle={self.steering_angle}, velocity={self.velocity}, yaw_angle={self.yaw_angle}, yaw_rate={self.yaw_rate}, slip_angle={self.slip_angle})"
    
    def to_list(self):
        return [self.x, self.y, self.steering_angle, self.velocity, self.yaw_angle, self.yaw_rate, self.slip_angle]

class MPCController:
    def __init__(self, dt=0.05, horizon=10, reference_trajectory=None):
        self.dt = dt  # Time step for MPC
        self.horizon = horizon  # MPC horizon
        self.p = parameters_vehicle1()  # Vehicle parameters
        self.p.l = 4.71932 #model3 length
        self.p.w = 2.09 #model3 width
        self.p.m = 1845
        self.p.Iz =  1500  ##0.95*p.m*(p.l/2)**2 # from https://github.com/carla-simulator/carla/issues/3221
        self.p.a = 1.5
        self.p.b = 1.5
        self.p.h_cg = 0.45
        self.target_speed = 30  # Reference speed
        self.wt1 = 100  # Weight for cte
        self.wt2 = 100  # Weight for epsi
        self.wt3 = 1  # Weight for speed error
        self.wt4 = 1  # Weight for actuations
        self.wt5 = 10  # Weight for actuation rate of change
        #reference trajectory is # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
        self.reference_trajectory = reference_trajectory if reference_trajectory is not None else np.zeros((horizon, 7))
        self.ref_line = LineString(self.reference_trajectory[:, 1:3])
        self.max_acceleration = 4.3  # Maximum acceleration
        self.acc_speed_intercept = 0.037

        self.max_steering = 70 * np.pi / 180  # Maximum steering angle

        self.last_predicted_actuation = np.array([0,0.9] * self.horizon)

        self.prev_x, self.prev_y, self.prev_yaw = 0, 0, 0
        self.prev_delta, self.prev_th = 0, 0
    
    def acceleration_from_throttle_and_speed(self, throttle, speed):
        return throttle * (self.max_acceleration - self.acc_speed_intercept*speed)

    def steering_from_delta_and_speed(self, delta, speed):
        return -delta*self.max_steering


    def mpc_cost_function(self, initial_state, control_inputs):
        assert len(control_inputs) == self.horizon * 2, f"Control inputs shape {control_inputs.shape} does not match horizon {self.horizon}"
        # Initialize cost
        total_cost = 0.0
        # Initialize state
        diff_angle = (initial_state.yaw_angle -self.prev_yaw)
        diff_angle = self.normalize_angle(diff_angle)
        yaw_rate = diff_angle/self.dt
        speed_vector = [initial_state.x - self.prev_x, initial_state.y - self.prev_y]
        norm = np.linalg.norm(speed_vector)
        if norm<1e-3:
            slip_angle = 0
            speed_vector = [0,0]
        else:
            speed_vector /= norm
            slip_angle = np.arctan2(speed_vector[1], speed_vector[0]) - initial_state.yaw_angle
            slip_angle = self.normalize_angle(slip_angle)
        steering_angle = self.steering_from_delta_and_speed(control_inputs[0], initial_state.velocity)
        state = State(x=initial_state.x, y=initial_state.y, steering_angle=steering_angle, velocity=initial_state.velocity, yaw_angle=initial_state.yaw_angle, yaw_rate=yaw_rate, slip_angle=slip_angle)
        
        prev_delta, prev_th = self.prev_delta, self.prev_th
        # Calculate cost for each time step
        for i in range(self.horizon):
            delta, th = control_inputs[2*i], control_inputs[2*i+1]
            # Compute cross track error (cte) and orientation error (epsi)
            cte, epsi, speed_error = self.compute_errors(state)
                        
            # Compute cost for each term
            cost_cte = self.wt1 * cte**2
            cost_epsi = self.wt2 * epsi**2
            cost_speed = self.wt3 * speed_error**2
            cost_actuations = self.wt4 * (th**2 + delta**2)
                        
            cost_steering_rate = self.wt5 * ((delta - prev_delta) / self.dt)**2
            cost_throttle_rate = self.wt5 * ((th - prev_th) / self.dt)**2

            prev_delta, prev_th = delta, th
            
            # Total cost for this time step
            total_cost += cost_cte + cost_epsi + cost_speed + cost_actuations + cost_steering_rate + cost_throttle_rate
            # Update state using bicycle model
            state = self.update_state(state, [delta, th])
            
        return total_cost

    def compute_errors(self, state):
        # Find the closest point on the reference trajectory to the vehicle
        x, y = state.x, state.y
        point = Point(x, y)
        proj = self.ref_line.project(point)
        closest_point = self.ref_line.interpolate(proj)
        cte = np.linalg.norm([x - closest_point.x, y - closest_point.y])

        # Compute orientation error by finding the angle between the vehicle orientation and the trajectory
        orientation = state.yaw_angle
        next_point = self.ref_line.interpolate(proj + 0.1)
        ref_orientation = np.arctan2(next_point.y - closest_point.y, next_point.x - closest_point.x)
        epsi = self.normalize_angle(orientation - ref_orientation)

        # Compute speed error
        idx_interpolated = bisect.bisect_left(self.reference_trajectory[:, 0], proj)
        alpha = (proj - self.reference_trajectory[idx_interpolated-1, 0])/(self.reference_trajectory[idx_interpolated, 0] - self.reference_trajectory[idx_interpolated-1, 0])
        interpolated_speed = (1-alpha)*self.reference_trajectory[idx_interpolated-1, 5] + alpha*self.reference_trajectory[idx_interpolated, 5]

        speed_error = interpolated_speed - state.velocity

        return cte, epsi, speed_error
    
    def update_state(self, state, control_input):
        # Update state using Single-Track (ST) model
        delta = control_input[0] # Steering control between -1 and 1
        th = control_input[1]  # Throttle/Brake
        a = self.acceleration_from_throttle_and_speed(th, state.velocity)
        steering_angle = self.steering_from_delta_and_speed(delta, state.velocity)

        state.steering_angle = steering_angle

        x0_ST = init_st(state.to_list())
        t = [0, self.dt]
        u = [0, a]
        
        pred = odeint(func_ST, x0_ST, t, args=(u, self.p))

        pred = pred[-1, :]

        # deriv = vehicle_dynamics_st(x0_ST, u, self.p) # Manual integration; can be used instead of odeint for speed but subject to unstable solutions
        # pred = np.array(x0_ST) + np.array(deriv) * self.dt

        state = State(x=pred[0], y=pred[1], steering_angle=steering_angle, velocity=pred[3], yaw_angle=pred[4], yaw_rate=pred[5], slip_angle=pred[6])
        state.yaw_angle = self.normalize_angle(state.yaw_angle)
        state.slip_angle = self.normalize_angle(state.slip_angle)
        self.prediction_time += self.dt

        return state
    
    def normalize_angle(self, angle):
        # Normalize angle between -pi and pi
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def solve_mpc(self, initial_state, current_time=None):
        self.current_time = current_time
        self.prediction_time = current_time 
        # Initial guess for control inputs using the last predicted actuation
        initial_guess = self.last_predicted_actuation
        #bounds are -1 to 1 for both steering and throttle
        bounds = np.array([(-1, 1)] * self.horizon * 2)
        assert len(initial_guess) == len(bounds), f"Initial guess shape {initial_guess.shape} does not match bounds shape {bounds.shape}"
        result = minimize(lambda x: self.mpc_cost_function(initial_state, x), initial_guess, bounds=bounds)
        optimal_control_inputs = result.x
        #return only the first control input
        self.last_predicted_actuation = optimal_control_inputs
        self.prev_x, self.prev_y, self.prev_yaw = initial_state.x, initial_state.y, initial_state.yaw_angle
        self.prev_delta, self.prev_th = optimal_control_inputs[0], optimal_control_inputs[1]
        return [optimal_control_inputs[0], optimal_control_inputs[1]]