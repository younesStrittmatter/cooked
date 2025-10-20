import numpy as np
import pandas as pd

def calculate_velocities(positions):
    velocities = np.diff(positions, axis=0, prepend=positions[0:1])
    return velocities

def load_positions(csv_path):
    df = pd.read_csv(csv_path)
    return df[['x', 'y']].values

def compute_scaler(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def apply_scaler(X, mu, sigma):
    return (X - mu) / sigma

def inverse_scaler(X_scaled, mu, sigma):
    return X_scaled * sigma + mu

def extract_features_targets(pos1, pos2):
    vel1 = calculate_velocities(pos1)
    vel2 = calculate_velocities(pos2)
    rel_pos = pos1 - pos2
    rel_vel = vel1 - vel2
    features = np.hstack([pos1, vel1, pos2, vel2, rel_pos, rel_vel])
    accel1 = np.diff(vel1, axis=0, prepend=vel1[0:1])
    accel2 = np.diff(vel2, axis=0, prepend=vel2[0:1])
    targets = np.hstack([accel1, accel2])
    return features, targets
