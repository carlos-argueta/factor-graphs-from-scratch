# Reference Extended Kalman Filter for the factor-graphs-from-scratch course.
# Copyright (c) 2026 Carlos Argueta — released under the MIT License (see LICENSE).

import numpy as np

import time

class ExtendedKalmanFilter:

	def __init__(self, initial_state, initial_covariance, motion_model, observation_model, **kwargs):
		# Process arguments
		proc_noise_std = kwargs.get('proc_noise_std', [0.02, 0.02, 0.01])
		obs_noise_std = kwargs.get('obs_noise_std', [0.02, 0.02, 0.01])

		self.mu = initial_state # Initial state estimate 
		self.Sigma = initial_covariance # Initial uncertainty

		self.g, self.G, self.V = motion_model() # The action model to use.
		
		# Standard deviations of the process or action model noise
		self.proc_noise_std = np.array(proc_noise_std)
		# Process noise covariance (R)
		self.R = np.diag(self.proc_noise_std ** 2)

		self.h, self.H = observation_model() # The observation model to use

		# Standard deviations for the observation or sensor model noise
		self.obs_noise_std = np.array(obs_noise_std)
		# Observation noise covariance (Q)
		self.Q = np.diag(self.obs_noise_std ** 2)

		# Per-call timing instrumentation is opt-in (debug=True); it prints on
		# every predict/update and accumulates unboundedly, so it is off in
		# production.
		self.debug = kwargs.get('debug', False)
		self.exec_times_pred = []
		self.exec_times_upd = []

		
	def predict(self, u, dt):
		start_time = time.time()
		# Predict state estimate (mu) 
		self.mu = self.g(self.mu, u, dt)
		# Predict covariance (Sigma)
		self.Sigma = self.G(self.mu, u, dt) @ self.Sigma @ self.G(self.mu, u, dt).T + self.R 

		if self.debug:
			execution_time = time.time() - start_time
			self.exec_times_pred.append(execution_time)
			print(f"Execution time prediction: {execution_time} seconds")
			print("Average exec time pred: ", sum(self.exec_times_pred) / len(self.exec_times_pred))

		return self.mu, self.Sigma

	def update(self, z, dt=None, h=None, H=None, Q=None):
		"""Correct the state with a measurement z.

		Pass h, H and Q to fuse an auxiliary sensor (e.g. GPS) as an
		independent sequential (partitioned) update; omit them to use
		the primary observation model given at construction time.
		"""
		start_time = time.time()

		h = h if h is not None else self.h
		H = H if H is not None else self.H
		Q = Q if Q is not None else self.Q

		# Evaluate the Jacobian once, at the prior mean
		H_mu = H(self.mu)

		# Compute the Kalman gain (K)
		S = H_mu @ self.Sigma @ H_mu.T + Q # Innovation covariance
		K = self.Sigma @ H_mu.T @ np.linalg.inv(S)

		# Update state estimate (mu)
		innovation = z - h(self.mu)
		self.mu = self.mu + (K @ innovation).reshape((self.mu.shape[0],))

		# Update covariance (Sigma)
		I = np.eye(self.mu.shape[0])
		self.Sigma = (I - K @ H_mu) @ self.Sigma

		if self.debug:
			execution_time = time.time() - start_time
			self.exec_times_upd.append(execution_time)
			print(f"Execution time update: {execution_time} seconds")
			print("Average exec time update: ", sum(self.exec_times_upd) / len(self.exec_times_upd))

		return self.mu, self.Sigma