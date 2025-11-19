from coppeliasim_zmqremoteapi import RemoteAPIClient
import time
import numpy as np

# conecta al servidor remoto de CoppeliaSim (debe estar corriendo)
client = RemoteAPIClient('localhost', 19997)
sim = client.getObject('sim')

deck = sim.getObjectHandle('deck_dummy')
uav = sim.getObjectHandle('uav_dummy')

sim.startSimulation()

# parámetros del movimiento del deck (barco)
t = 0.0
dt = 0.05
A, w = 0.3, 0.8  # amplitud y frecuencia del oleaje
vx, vy = 0.05, 0.02

# UAV simple control hacia deck
kp = 0.8  # ganancia proporcional

for step in range(400):
    # posición del deck: simulamos ola + deriva
    x = vx * t
    y = vy * t
    z = 0.2 * np.sin(w * t)
    sim.setObjectPosition(deck, -1, [x, y, z])

    # obtener posición UAV
    uav_pos = sim.getObjectPosition(uav, -1)
    deck_pos = np.array([x, y, z + 0.3])  # punto deseado sobre deck

    # error UAV->deck
    err = deck_pos - np.array(uav_pos)
    vel_cmd = kp * err  # velocidad proporcional

    # limitar velocidad
    max_vel = 0.2
    norm = np.linalg.norm(vel_cmd)
    if norm > max_vel:
        vel_cmd *= max_vel / norm

    # mover UAV
    new_pos = np.array(uav_pos) + vel_cmd * dt
    sim.setObjectPosition(uav, -1, new_pos.tolist())

    t += dt
    time.sleep(dt)

sim.stopSimulation()
