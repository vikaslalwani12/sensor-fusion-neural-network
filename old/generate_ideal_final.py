import os
import sys
import traci
import math
import csv

# SUMO setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

sumo_cmd = ["sumo", "-c", "simulation.sumocfg"]
traci.start(sumo_cmd)

# Camera scaling parameter
d0 = 30  # meters (city scenario)

with open("dataset_ideal_final.csv", "w", newline="") as file:
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

            # FEATURES
            relative_speed = abs(ego_speed - closest_speed)

            # Updated camera model
            camera = 1 / (1 + (min_dist / d0)**2)

            ultrasonic = 1 if min_dist < 5 else 0

            # Updated risk model
            risk_score = camera + (relative_speed / 30)

            writer.writerow([
                min_dist,
                relative_speed,
                closest_angle,
                camera,
                ultrasonic,
                risk_score
            ])

traci.close()
print("Ideal dataset generated (final).")