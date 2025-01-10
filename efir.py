# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:04:42 2024

@author: Florent Boxus
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
np.random.seed(24)

# Define known landmark positions
x1, y1 = 0.0, 20.0  # Landmark 1
x2, y2 = 0.0, 0.0  # Landmark 2
x3, y3 = 30.0, 0.0  # Landmark 3

# Parameters
Q = np.diag([1e-4, 1e-4, 7.62e-5])#noise on state
L = np.diag([1e-4, 1e-4])  # noise for incremental movement

sigma_phi1 = np.deg2rad(0.5)  # angle measurement noise
R = np.diag([1.218e-3, 1.218e-3, 1.218e-3])

b=0.2#distance between both wheels /!\ I decided myself for this value

# Noise scaling factor for "correcting" the process noise
p = 5
Q *= p**2
R /= p**2
L /= p**2

# Initial robot state (position, heading)
x0 = np.array([0.0, 0.0, 0.0])  # initial position at origin and 0 heading
P0 = np.eye(3)  # initial estimate covariance

def get_d_n(dl,dr):
    return 0.5*(dl+dr)
def get_phi_n(dl,dr):
    return math.atan((dr-dl)/b)


# Robot movement and measurement model
def motion_model(xn_minus_1, un, noise):
    """ Robot motion model """
    x, y, phi = xn_minus_1
    dL, dR = un
    dtheta = (dR - dL) / b
    dxy = (dL + dR) / 2.0
    

    xn = np.array([x + dxy * np.cos(phi + dtheta/2),
                   y + dxy * np.sin(phi + dtheta/2),
                   phi + dtheta])
    if noise is not None:
        xn += noise
    return xn

def observation_model(xn):
    """ Robot observation model (angles) """
    x, y, phi = xn  # Robot's state: [x position, y position, heading]


    # Compute relative positions
    Qin1, Iin1 = y1 - y, x1 - x
    Qin2, Iin2 = y2 - y, x2 - x
    Qin3, Iin3 = y3 - y, x3 - x

    # Calculate the angles based on the condition for Qin > 0 and Qin < 0
    def calc_theta(Qin, Iin):
        if Iin >= 0:
            return np.arctan2(Qin, Iin)
        elif Iin < 0:
            return np.arctan2(Qin, Iin) + np.pi

    # Calculate the measured angles
    theta1 = calc_theta(Qin1, Iin1)
    theta2 = calc_theta(Qin2, Iin2)
    theta3 = calc_theta(Qin3, Iin3)

    # The angles measured are phi_i = theta_i - phi
    phi1 = theta1 - phi
    phi2 = theta2 - phi
    phi3 = theta3 - phi

    # Return the angles (modulo 2π to ensure they are within the valid range)
    return np.array([(phi1+np.pi) % (2 * np.pi)-np.pi,( phi2+np.pi) % (2 * np.pi)-np.pi, (phi3+np.pi) % (2 * np.pi)-np.pi])
def linear_position(phi_array):
    phi1=phi_array[0]
    phi2=phi_array[1]
    phi3=phi_array[2]
    A=(y1/x3)*(np.sin(phi3-phi2)/np.sin(phi2-phi1))
    phi=math.atan((A*np.cos(phi1)-np.sin(phi3))/(A*np.sin(phi1)+np.cos(phi3)))
    x=(x3*np.tan(phi3+phi))/(np.tan(phi3+phi)-np.tan(phi2+phi))
    y=x*np.tan(phi2+phi)
    return [x,y,phi]
# Jacobian of the motion model F_n
def F_n(xn_minus_1, un):
    x, y, phi = xn_minus_1
    dL, dR = un
    d_n=get_d_n(dL, dR)
    phi_n=get_phi_n(dL, dR)
    F = np.array([[1, 0, -d_n * np.sin(phi + (phi_n )/ 2)],
                  [0, 1, d_n * np.cos(phi + (phi_n )/ 2)],
                  [0, 0, 1]])
    return F
def E_n(xn_minus_1, un):
    x, y, phi = xn_minus_1
    dL, dR = un
    dn=get_d_n(dL, dR)
    phi_n=get_phi_n(dL, dR)
    E = np.array([
    [0.5 * (math.cos(phi + phi_n / 2)) + (dn / (2 * b)) * math.sin(phi + phi_n / 2), 0.5 * (math.cos(phi + phi_n / 2)) - (dn / (2 * b)) * math.sin(phi + phi_n / 2)],
    [0.5 * (math.sin(phi + phi_n / 2)) - (dn / (2 * b)) * math.cos(phi + phi_n / 2), 0.5 * (math.sin(phi + phi_n / 2)) + (dn / (2 * b)) * math.cos(phi + phi_n / 2)],
    [-1 / b, 1 / b]]
    )
    return E
# Jacobian of the observation model H_n
def H_n(x_pred):
    x, y, phi = x_pred
    H = np.array([[(y1-y)/(x**2+(y1-y)**2),(x)/(x**2+(y1-y)**2),-1],
                  [(-y)/(x**2+y**2),(x)/(x**2+y**2),-1],
                  [(-y)/((x3-x)**2+y**2),(-x3+x)/((x3-x)**2+y**2),-1]])
    return H

# Extended Kalman Filter (EKF) Prediction Step
def ekf_predict(xn_minus_1, Pn_minus_1, un):
    # Prédiction de l'état
    xn_pred = motion_model(xn_minus_1, un, None)
    
    # Calcul des Jacobiennes
    F = F_n(xn_minus_1, un)
    E = E_n(xn_minus_1, un)
    
    # Prédiction de la covariance de l'état
    Pn_pred = np.dot(F, np.dot(Pn_minus_1, F.T)) + np.dot(F, np.dot(Q, F.T)) + np.dot(E, np.dot(L, E.T))
    
    return xn_pred, Pn_pred

# EKF Update Step
def ekf_update(xn_pred, Pn_pred, zn):
    # Jacobienne de l'observation
    H = H_n(xn_pred)
    
    # Innovation
    y_n = zn - observation_model(xn_pred)
    
    # Calcul de la covariance d'innovation
    S_n = np.dot(H, np.dot(Pn_pred, H.T)) + R
    
    # Gain de Kalman
    K_n = np.dot(Pn_pred, np.dot(H.T, np.linalg.inv(S_n)))
    
    # Mise à jour de l'état
    xn_updated = xn_pred + np.dot(K_n, y_n)
    
    # Mise à jour de la covariance de l'état
    Pn_updated = np.dot(np.eye(len(xn_pred)) - np.dot(K_n, H), Pn_pred)
    
    return xn_updated, Pn_updated
# Simulating robot motion
def simulate_and_plot(N, plot):
    num_steps = 1000
    state_history = []
    measurement_history = []
    estimated_state_history = []
    estimated_cov_history = []
    linear_measure = []
    observation_history = []
    estimated_ekf_state=[]
    command_history=[]
    # Initial state
    xn_true = x0
    Pn_true = P0
    xn_est = x0
    xn_est2=xn_est.copy()
    Pn_est = P0
    Pn_est2=P0
    K = 3
    G = np.zeros((num_steps, 3, 3))
    F = G.copy()
    x_tilde=np.zeros((num_steps,3))

    # Simulate robot movement and apply EKF
    for step in range(num_steps):
        # Simulate noise in motion and measurement
        noise_motion = np.random.multivariate_normal([0, 0, 0], Q)  # state noise
        noise_measurement = np.random.multivariate_normal([0, 0, 0], R)  # measurement noise
        
        # Incremental movement (dL, dR)
        dL = 1.0 + np.random.normal(0, 1e-2)  
        dR = 1.0 + np.random.normal(0, 1e-2)  
        command_history.append([dL,dR])
        # True motion update
        xn_true = motion_model(xn_true, [dL, dR], noise=noise_motion)
        
        if step < 1:
            F[step] = np.eye(3)
        else:
            F[step] = F_n(estimated_state_history[step-1], [dL, dR])
        
        # Simulate measurements (with noise)
        zn_true = observation_model(xn_true) + noise_measurement
        linear_measure.append(linear_position(zn_true))
        observation_history.append(zn_true)
        xn_pred2, Pn_pred2 = ekf_predict(xn_est2, Pn_est2, [dL, dR])
        xn_est2, Pn_est2 = ekf_update(xn_pred2, Pn_pred2, zn_true)
        estimated_ekf_state.append(xn_est2)
        if step < N:
            xn_pred, Pn_pred = ekf_predict(xn_est, Pn_est, [dL, dR])
            xn_est, Pn_est = ekf_update(xn_pred, Pn_pred, zn_true)
        else:
            m = step - N + 1
            s = m + K - 1


            if s < N-1:
                x_tilde[s] = linear_measure[s] 
            else:
                x_tilde[s] = estimated_state_history[s]

            G[s]=np.eye(3)
            for l in range(m+K, step+1,1):
                x_pred = motion_model(x_tilde[l-1], command_history[l], noise=None)
                H = H_n(x_pred)
                O=np.linalg.inv(F[l] @ G[l-1] @ F[l].T)
                G[l] = np.linalg.pinv(H.T @ H + O)
                Kl = G[l] @ H.T
                x_tilde[l] = x_pred + Kl @ (observation_history[l] - observation_model(x_pred))
                
            xn_est = x_tilde[step]  # Final estimate
            Pn_est = np.outer(xn_true - xn_est, xn_true - xn_est)  # Covariance update

        
        state_history.append(xn_true)
        measurement_history.append(zn_true)
        estimated_state_history.append(xn_est)

        estimated_cov_history.append(Pn_est)


    # Convert lists to arrays for plotting
    state_history = np.array(state_history)
    estimated_state_history = np.array(estimated_state_history)
    estimated_ekf_state=np.array(estimated_ekf_state)
    pn=(state_history-estimated_state_history)@(state_history-estimated_state_history).T
    print('Trace de la matrice:',np.trace(pn))
    linear_measure = np.array(linear_measure)
    if plot==False:
        return np.trace(pn)
    else:
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(state_history[:, 0], state_history[:, 1], label="True Robot Path")
        plt.plot(estimated_state_history[:, 0], estimated_state_history[:, 1], label="EFIR", linestyle="--")
        plt.plot(estimated_ekf_state[:,0],estimated_ekf_state[:,1],label='EKF',color='green')
        #plt.plot(linear_measure[:,0],linear_measure[:,1],label="Linear estimation")
        plt.scatter([x1, x2, x3], [y1, y2, y3], c='red', marker='x', s=100, label="Beacons")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend()
        plt.title("Mobile Robot Path: True vs EKF Estimates")
        plt.grid(True)
        plt.savefig("robot_path_efir_noise.jpeg",  format='jpeg', dpi=1000)
        plt.show()
        
        error_efir = np.linalg.norm(state_history - estimated_state_history, axis=1)
        error_ekf = np.linalg.norm(state_history - estimated_ekf_state, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(error_efir, label="Erreur EFIR", linestyle="--", color="blue")
        plt.plot(error_ekf, label="Erreur EKF", linestyle="--", color="green")
        plt.xlabel("Indice temporel (étape)")
        plt.ylabel("Erreur (distance euclidienne)")
        plt.title("Comparaison des erreurs: EFIR vs EKF")
        plt.legend()
        plt.grid(True)
        plt.savefig("error_comparison_noise.jpeg", format='jpeg', dpi=1000)
        plt.show()
        
        # Calculer les erreurs pour chaque dimension
        error_efir_x = state_history[:, 0] - estimated_state_history[:, 0]
        error_efir_y = state_history[:, 1] - estimated_state_history[:, 1]
        error_ekf_x = state_history[:, 0] - estimated_ekf_state[:, 0]
        error_ekf_y = state_history[:, 1] - estimated_ekf_state[:, 1]
        
        # Créer la figure pour les erreurs en X
        plt.figure(figsize=(10, 6))
        plt.plot(error_efir_x, label="Erreur EFIR en X", linestyle="--", color="blue")
        plt.plot(error_ekf_x, label="Erreur EKF en X", linestyle="--", color="green")
        plt.xlabel("Indice temporel (étape)")
        plt.ylabel("Erreur en X (m)")
        plt.title("Comparaison des erreurs en X : EFIR vs EKF")
        plt.legend()
        plt.grid(True)
        plt.savefig("error_comparison_x_noise.jpeg", format='jpeg', dpi=1000)
        plt.show()

        # Créer la figure pour les erreurs en Y
        plt.figure(figsize=(10, 6))
        plt.plot(error_efir_y, label="Erreur EFIR en Y", linestyle="--", color="blue")
        plt.plot(error_ekf_y, label="Erreur EKF en Y", linestyle="--", color="green")
        plt.xlabel("Indice temporel (étape)")
        plt.ylabel("Erreur en Y (m)")
        plt.title("Comparaison des erreurs en Y : EFIR vs EKF")
        plt.legend()
        plt.grid(True)
        plt.savefig("error_comparison_y_noise.jpeg", format='jpeg', dpi=1000)
        plt.show()

minN=float('inf')
bestN=0
for N in range(5,20,1):
    result=simulate_and_plot(N, False)
    print(N)
    if(result<minN):
        bestN=N
        minN=result
print('N optimal:',bestN)
simulate_and_plot(bestN,True)
    
    