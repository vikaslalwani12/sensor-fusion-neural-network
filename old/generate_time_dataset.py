import os
import sys
import traci
import math
import csv
import random

# -------- SUMO PATH --------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare SUMO_HOME")

# -------- PARAMETERS --------
TARGET_ROWS = 50000   # 🔥 dataset size target

# -------- FUNCTIONS --------

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def camera_confidence(d):
    return 1 / (1 + (d / 40)**2)

def risk_score(dist, rel_speed, cam, ultrasonic):

    # Distance risk (strong nonlinear)
    if dist < 5:
        dist_risk = 1.0
    elif dist < 10:
        dist_risk = 0.8
    elif dist < 20:
        dist_risk = 0.5
    elif dist < 40:
        dist_risk = 0.2
    else:
        dist_risk = 0.05

    # Speed risk (more aggressive)
    speed_risk = min((rel_speed / 12)**1.2, 1)

    # Camera uncertainty
    cam_risk = (1 - cam)**1.2

    # Ultrasonic boost
    ultra_risk = ultrasonic * 1.0

    # Final weighted
    risk = (
        0.55 * dist_risk +
        0.2 * speed_risk +
        0.15 * cam_risk +
        0.1 * ultra_risk
    )

    return round(min(risk, 1.0), 3)

# -------- FILE --------
file = open("dataset_large_50k.csv", "w", newline="")
writer = csv.writer(file)

writer.writerow([
    "time",
    "distance",
    "relative_speed",
    "camera",
    "ultrasonic",
    "angle",
    "risk_score"
])

row_count = 0
run = 0

print("\nGenerating dataset...\n")

# -------- MAIN LOOP --------
while row_count < TARGET_ROWS:
    run += 1
    print(f"Run {run} started...")

    # Start SUMO
    traci.start(["sumo", "-c", "simulation.sumocfg"])

    ego_vehicle = None
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()
        step += 1

        vehicles = traci.vehicle.getIDList()

        if len(vehicles) < 2:
            continue

        # Ego selection
        if ego_vehicle is None or ego_vehicle not in vehicles:
            ego_vehicle = vehicles[0]

        ego_pos = traci.vehicle.getPosition(ego_vehicle)
        ego_speed = traci.vehicle.getSpeed(ego_vehicle)

        min_dist = float('inf')
        nearest_vehicle = None

        # Find nearest vehicle
        for v in vehicles:
            if v == ego_vehicle:
                continue

            pos = traci.vehicle.getPosition(v)
            d = distance(ego_pos, pos)

            if d < min_dist:
                min_dist = d
                nearest_vehicle = v

        if nearest_vehicle is None:
            continue

        other_speed = traci.vehicle.getSpeed(nearest_vehicle)

        # -------- FEATURES --------
        dist = min_dist
        rel_speed = abs(ego_speed - other_speed)

        cam = camera_confidence(dist)
        cam += random.uniform(-0.02, 0.02)
        cam = max(0, min(1, cam))

        ultrasonic = 1 if dist < 5 else 0
        angle = random.uniform(-3.14, 3.14)

        risk = risk_score(dist, rel_speed, cam, ultrasonic)

        # -------- SAVE --------
        writer.writerow([step + run*10000, dist, rel_speed, cam, ultrasonic, angle, risk])

        row_count += 1

        if row_count % 5000 == 0:
            print(f"Rows generated: {row_count}")

        # STOP if target reached
        if row_count >= TARGET_ROWS:
            break

    # Close SUMO safely
    try:
        traci.close(False)
    except:
        pass

print("\nDataset generation complete!")
print(f"Total rows: {row_count}")

file.close()