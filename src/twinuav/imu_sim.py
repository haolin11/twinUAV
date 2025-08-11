from __future__ import annotations
import numpy as np
import math

class IMUSim:
    def __init__(self, gyro_noise=0.005, accel_noise=0.02, gyro_bias_drift=1e-4, accel_bias_drift=1e-3, g=9.81):
        self.gyro_noise = gyro_noise
        self.accel_noise = accel_noise
        self.gyro_bias = np.zeros(3, dtype=np.float32)
        self.accel_bias = np.zeros(3, dtype=np.float32)
        self.gyro_bias_drift = gyro_bias_drift
        self.accel_bias_drift = accel_bias_drift
        self.g = g
    def step(self, v_fwd: float, yaw_rate: float, yaw: float):
        gyro = np.array([0.0, 0.0, yaw_rate], dtype=np.float32)
        gyro += self.gyro_bias + np.random.randn(3).astype(np.float32) * self.gyro_noise
        g_world = np.array([0.0, 0.0, -self.g], dtype=np.float32)
        R_wb = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0,0.0,1.0]], dtype=np.float32)
        a_body = R_wb.T @ g_world
        accel = -a_body
        accel += self.accel_bias + np.random.randn(3).astype(np.float32) * self.accel_noise
        self.gyro_bias += np.random.randn(3).astype(np.float32) * self.gyro_bias_drift
        self.accel_bias += np.random.randn(3).astype(np.float32) * self.accel_bias_drift
        return gyro, accel