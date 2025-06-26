import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load environment and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# Create cube (visual + collision)
cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1])
cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
body_id = p.createMultiBody(baseMass=1,
                            baseCollisionShapeIndex=cube_collision,
                            baseVisualShapeIndex=cube_visual,
                            basePosition=[0.6, 0, 0.05])

# Let environment settle
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./30.)

# Logs
joint_log = []
ee_log = []
action_log = []

def move_arm_to(position, steps=200, label=""):
    joint_angles = p.calculateInverseKinematics(robot_id, 6, position)
    for i in range(6):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_angles[i])
    for _ in range(steps):
        log_state()
        p.stepSimulation()
        time.sleep(1./120.)

    # Log each motion step
    ee_pos = p.getLinkState(robot_id, 6)[0]
    action_log.append({
        "Action": label,
        "Steps": steps,
        "Target X": round(position[0], 3),
        "Target Y": round(position[1], 3),
        "Target Z": round(position[2], 3),
        "Final X": round(ee_pos[0], 3),
        "Final Y": round(ee_pos[1], 3),
        "Final Z": round(ee_pos[2], 3),
    })

def log_state():
    joint_states = [p.getJointState(robot_id, i)[0] for i in range(6)]
    ee_pos = p.getLinkState(robot_id, 6)[0]
    joint_log.append(joint_states)
    ee_log.append(ee_pos)

# Task setup
pick_pos = [0.6, 0, 0.05]
place_pos = [0.3, -0.3, 0.05]
hover_height = 0.3

# Pick-and-place sequence
move_arm_to([pick_pos[0], pick_pos[1], hover_height], label="Hover over pick")
move_arm_to(pick_pos, label="Move to pick")

# Create fixed constraint to simulate grasp
grasp_constraint = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=6,
    childBodyUniqueId=body_id,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)
print(f"Cube picked at: {pick_pos[0]}, {pick_pos[1]}, {hover_height}")

move_arm_to([pick_pos[0], pick_pos[1], hover_height], label="Lift after pick")
move_arm_to([place_pos[0], place_pos[1], hover_height], label="Hover over place")
move_arm_to(place_pos, label="Move to place")

# Remove constraint to simulate release
p.removeConstraint(grasp_constraint)
print(f"Cube placed at: {place_pos[0]}, {place_pos[1]}, {place_pos[2]}")

move_arm_to([place_pos[0], place_pos[1], hover_height], label="Lift after place")

# Final hold
for _ in range(200):
    log_state()
    p.stepSimulation()
    time.sleep(1./120.)

p.disconnect()

# Convert logs
joint_log = np.array(joint_log)
ee_log = np.array(ee_log)

# Plot joint angles
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot(joint_log[:, i], label=f'Joint {i}')
plt.title('Joint Angles Over Time')
plt.xlabel('Timestep')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot end-effector position
plt.figure(figsize=(10, 5))
plt.plot(ee_log[:, 0], label='X')
plt.plot(ee_log[:, 1], label='Y')
plt.plot(ee_log[:, 2], label='Z')
plt.title('End-Effector Position Over Time')
plt.xlabel('Timestep')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary table
print("\nPick-and-Place Summary")
df = pd.DataFrame(action_log)
print(df.to_string(index=False))
