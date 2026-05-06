import random

input_file = "routes.rou.xml"
output_file = "routes_fixed.rou.xml"

types = ["slow", "normal", "fast"]

with open(input_file, "r") as f:
    lines = f.readlines()

new_lines = []
added_vtypes = False

for line in lines:

    # Add vTypes once
    if "<routes" in line and not added_vtypes:
        new_lines.append(line)
        new_lines.append('    <vType id="slow" accel="1.0" decel="2.0" maxSpeed="12" sigma="1.0"/>\n')
        new_lines.append('    <vType id="normal" accel="2.5" decel="4.5" maxSpeed="18" sigma="0.7"/>\n')
        new_lines.append('    <vType id="fast" accel="4.0" decel="6.0" maxSpeed="30" sigma="0.4"/>\n')
        added_vtypes = True
        continue

    if "<vehicle" in line:
        vtype = random.choices(types, weights=[0.3, 0.5, 0.2])[0]
        line = line.replace("<vehicle ", f'<vehicle type="{vtype}" ')

    new_lines.append(line)

with open(output_file, "w") as f:
    f.writelines(new_lines)

print("✅ routes_fixed.rou.xml created")