"""Render the Unitree Go2 (MuJoCo Menagerie) into a segmentation + depth mask
for the particle header. Usage: python go2_mask.py model.xml out.npz [azimuth elevation]"""
import sys
import numpy as np
import mujoco

xml, out = sys.argv[1], sys.argv[2]
az = float(sys.argv[3]) if len(sys.argv) > 3 else 140
el = float(sys.argv[4]) if len(sys.argv) > 4 else -12
W, H = 1100, 312

m = mujoco.MjModel.from_xml_path(xml)
m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
m.vis.global_.offheight = max(m.vis.global_.offheight, H)
d = mujoco.MjData(m)
if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

r = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = d.qpos[:3] + np.array([0.0, 0.0, -0.12])
cam.distance = 0.74
cam.azimuth, cam.elevation = az, el

r.enable_segmentation_rendering()
r.update_scene(d, cam)
seg = r.render()[..., 0].copy()          # geom id per pixel, -1 = background
r.disable_segmentation_rendering()
r.enable_depth_rendering()
r.update_scene(d, cam)
depth = r.render().copy()
r.disable_depth_rendering()

# keep only geoms belonging to the robot (not floor / terrain)
body_of = m.geom_bodyid
robot = np.zeros_like(seg, bool)
for g in np.unique(seg):
    if g < 0:
        continue
    if body_of[g] != 0:                  # body 0 = world (floor)
        robot |= seg == g
seg = np.where(robot, seg, -1)
print("robot pixels:", robot.sum(), "bbox:", np.argwhere(robot).min(0), np.argwhere(robot).max(0))
np.savez_compressed(out, seg=seg, depth=depth.astype(np.float32))
