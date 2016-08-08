import numpy as np

# Implements a linear Kalman Filter
class KalmanFilterLinear:
    def __init__(self, A, B, H, x, P, Q, R):
        self.A = A                          # state transition matrix
        self.B = B                          # control matrix
        self.H = H                          # observation matrix
        self.current_state_estimate = x     # Initial state estimate
        self.current_prob_estimate = P      # Initial covariance estimate
        self.Q = Q                          # Estimated error in process
        self.R = R                          # Estimated error in measurements

    def GetCurrentState(self):
        return self.current_state_estimate

    def Step(self, control_vector, measurement_vector):
        #---------------------Prediction Step----------------------------
        predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_vector
        predicted_prob_estimate = (self.A * self.current_prob_estimate) * np.transpose(self.A) + self.Q

        #---------------------Observation step---------------------------
        innovation = measurement_vector - self.H * predicted_state_estimate
        innovation_covariance = self.H * predicted_prob_estimate * np.transpose(self.H) + self.R

        #---------------------Update step--------------------------------
        kalman_gain = predicted_prob_estimate * np.transpose(self.H) * np.linalg.inv(innovation_covariance)
        self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation

        # We need the size of the matrix so we can make an identity matrix.
        size = self.current_prob_estimate.shape[0]

        # eye(N) = NxN identity matrix
        self.current_prob_estimate = (np.eye(size)-kalman_gain*self.H)*predicted_prob_estimate