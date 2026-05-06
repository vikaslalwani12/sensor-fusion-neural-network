import os
import sys
import traci
import math
import csv
import random

# Add SUMO tools path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare SUMO_HOME")

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

sumo_cmd = ["sumo", "-c", "simulation.sumocfg"]
traci.start(sumo_cmd)

print("Generating realistic dataset...")

try:
    with open("dataset_realistic.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["distance", "camera", "ultrasonic"])

        for step in range(5000):
            traci.simulationStep()

            vehicles = traci.vehicle.getIDList()

            if len(vehicles) < 2:
                continue

            for ego in vehicles:

                ego_pos = traci.vehicle.getPosition(ego)

                min_dist = float('inf')

                for v in vehicles:
                    if v == ego:
                        continue

                    pos = traci.vehicle.getPosition(v)
                    d = distance(ego_pos, pos)

                    if d < min_dist:
                        min_dist = d

                # -------- REALISM STARTS HERE --------

                # 1. Radar noise (Gaussian noise ±5%)
                noise = random.uniform(-0.05, 0.05) * min_dist
                radar = max(min_dist + noise, 0)

                # 2. Camera imperfection (nonlinear + noise)
                camera = 1 / (1 + radar)
                camera += random.uniform(-0.02, 0.02)
                camera = max(min(camera, 1), 0)

                # 3. Ultrasonic with false positives/negatives
                if radar < 5:
                    ultrasonic = 1 if random.random() > 0.1 else 0   # 10% miss
                else:
                    ultrasonic = 1 if random.random() < 0.05 else 0  # 5% false detect

                writer.writerow([radar, camera, ultrasonic])

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    try:
        traci.close()
    except:
        pass

print("Realistic dataset generated.")