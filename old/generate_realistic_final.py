import os
import sys
import traci
import math
import csv
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

sumo_cmd = ["sumo", "-c", "simulation.sumocfg"]
traci.start(sumo_cmd)

d0 = 30  # camera range scaling

with open("dataset_realistic_final.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "distance",
        "relative_speed",
        "angle",
        "camera",
        "ultrasonic",
        "risk_score"
    ])

    for step in range(5000):
        traci.simulationStep()

        vehicles = traci.vehicle.getIDList()
        if len(vehicles) < 2:
            continue

        for ego in vehicles:

            ego_pos = traci.vehicle.getPosition(ego)
            ego_speed = traci.vehicle.getSpeed(ego)

            min_dist = float('inf')
            closest_speed = 0
            closest_angle = 0

            for v in vehicles:
                if v == ego:
                    continue

                pos = traci.vehicle.getPosition(v)
                d = distance(ego_pos, pos)

                if d < min_dist:
                    min_dist = d
                    closest_speed = traci.vehicle.getSpeed(v)

                    dx = pos[0] - ego_pos[0]
                    dy = pos[1] - ego_pos[1]
                    closest_angle = math.atan2(dy, dx)

            # -------- REALISTIC MODEL --------

            # Distance noise (±5%)
            dist = max(min_dist + random.uniform(-0.05, 0.05)*min_dist, 0)

            # Relative speed noise
            relative_speed = abs(ego_speed - closest_speed)
            relative_speed += random.uniform(-2, 2)

            # Camera with noise
            camera = 1 / (1 + (dist / d0)**2)
            camera += random.uniform(-0.03, 0.03)
            camera = max(min(camera, 1), 0)

            # Ultrasonic with noise
            if dist < 5:
                ultrasonic = 1 if random.random() > 0.1 else 0
            else:
                ultrasonic = 1 if random.random() < 0.05 else 0

            # Risk score
            risk_score = camera + (relative_speed / 30)

            writer.writerow([
                dist,
                relative_speed,
                closest_angle,
                camera,
                ultrasonic,
                risk_score
            ])

traci.close()
print("Realistic dataset generated (final).")