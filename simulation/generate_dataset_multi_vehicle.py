import os
import sys
import traci
import math
import csv
import random

# ================== SUMO SETUP ==================
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare SUMO_HOME")

# ================== FUNCTIONS ==================
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def camera_conf(d):
    # realistic-ish decay
    return max(0.25, 1 / (1 + 0.02 * d))

def compute_risk(d, rel_speed, cam, ultra):
    # balanced risk model
    risk = (
        0.45*(1/(d+1)) +
        0.4*(rel_speed/20) +
        0.1*(1 - cam) +
        0.05*ultra
    )
    return min(max(risk, 0), 1)

# ================== START SUMO ==================
sumo_cmd = ["sumo", "-c", "simulation.sumocfg"]
traci.start(sumo_cmd)

# ================== DATA FILE ==================
# Create data directory if it doesn't exist
os.makedirs("../data", exist_ok=True)
file = open("../data/dataset_final.csv", "w", newline="")
writer = csv.writer(file)

writer.writerow([
    "time","ego_id","target_id","distance",
    "relative_speed","camera","ultrasonic","angle","risk_score"
])

TARGET_ROWS = 50000
MAX_STEPS = 4000

rows = 0
step = 0

# ================== MAIN LOOP ==================
try:
    while rows < TARGET_ROWS:

        traci.simulationStep()
        step += 1

        vehicles = traci.vehicle.getIDList()

        # need enough vehicles for interactions
        if len(vehicles) < 6:
            continue

        # ================= SPEED RANDOMIZATION =================
        for v in vehicles:
            try:
                traci.vehicle.setSpeedMode(v, 0)

                speed = random.uniform(5, 25)
                speed += random.uniform(-3, 3)

                speed = max(0, min(speed, 30))
                traci.vehicle.setSpeed(v, speed)

            except:
                pass

        # ================= MULTI-EGO SAMPLING =================
        egos = random.sample(vehicles, min(6, len(vehicles)))

        for ego in egos:

            ego_pos = traci.vehicle.getPosition(ego)
            ego_speed = traci.vehicle.getSpeed(ego)

            targets = random.sample(
                [v for v in vehicles if v != ego],
                min(20, len(vehicles)-1)
            )

            for v in targets:

                if rows >= TARGET_ROWS:
                    break

                try:
                    pos = traci.vehicle.getPosition(v)
                    speed = traci.vehicle.getSpeed(v)
                except:
                    continue

                d = distance(ego_pos, pos)

                # keep realistic interaction range
                if d > 80:
                    continue

                rel_speed = abs(ego_speed - speed)

                cam = camera_conf(d)
                ultra = 1 if d < random.uniform(5, 10) else 0

                dx = pos[0] - ego_pos[0]
                dy = pos[1] - ego_pos[1]
                ang = math.atan2(dy, dx)

                risk = compute_risk(d, rel_speed, cam, ultra)

                writer.writerow([
                    step,
                    ego,
                    v,
                    round(d, 2),
                    round(rel_speed, 2),
                    round(cam, 3),
                    ultra,
                    round(ang, 3),
                    round(risk, 3)
                ])

                rows += 1

        if step % 50 == 0:
            print(f"Step {step} | Rows: {rows}")

        # safety stop
        if step >= MAX_STEPS:
            print("Reached max steps")
            break

except KeyboardInterrupt:
    print("Stopped manually")

finally:
    file.close()
    try:
        traci.close()
    except:
        pass

print(f"\n✅ DONE: {rows} rows generated")