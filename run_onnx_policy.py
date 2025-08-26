"""
Run a transferred PPO policy (ONNX) in MuJoCo with deterministic or stochastic actions,
a MuJoCo-derived state feature vector, and robust actuator-to-action mapping.

Notes:
- The ONNX was exported from ManiSkill PPO baseline and returns deterministic action_mean and value.
- Training likely used include_state=True. This runner now computes a state vector from MuJoCo.
- Stochastic policy uses a user-provided Gaussian std since the learned logstd wasn't exported.
"""

import argparse
import os
import time
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import mujoco as mj
from mujoco import viewer
import onnxruntime as ort

from robot_descriptions import panda_mj_description

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_ACTION_SCALE = 0.0005   # radians/step for arm; tune to your model/control setup
DEFAULT_CTRL_HZ = 50
DEFAULT_CAMERA_NAME = "sensor_cam"
DEFAULT_RENDER_WIDTH = 128
DEFAULT_RENDER_HEIGHT = 128

# Scene configs
cube_half_size = 0.02
goal_thresh = 0.025
cube_spawn_half_size = 0.1
max_goal_height = 0.3

MJCF_PATH: str = panda_mj_description._path.join(panda_mj_description.PACKAGE_PATH, "mjx_panda.xml")


def build_scene_xml(camera_name: str) -> str:
    return f"""
<mujoco model="tabletop_scene">
    <!-- Minimal world with table, cube, goal, camera, lights -->
    <asset>
        <model name="table_model" file="assets/table/table.xml"/>
        <model name="franka_panda_model" file="{MJCF_PATH}"/>

        <texture name="grid" type="2d" builtin="checker" rgb1=".5 .5 .5"
            rgb2=".5 .5 .5" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
    </asset>

    <worldbody>
        <light name="top" pos="-0.5 -0.5 2" castshadow="false" type="directional"/>

        <camera name="{camera_name}" pos="0.3 0 0.6" xyaxes="0 1 0 -0.5 0 0.4" fovy="90"/>

        <body name="table" pos="-0.12 0 -0.9196429">
            <attach model="table_model" prefix="table_"/>
        </body>

        <body name="franka_panda" pos="-0.615 0 0">
            <attach model="franka_panda_model" prefix="franka_panda_"/>
        </body>

        <body name="cube" pos="0 0 0">
            <geom name="cube" type="box" size="{cube_half_size} {cube_half_size} {cube_half_size}" rgba="1 0 0 1" />
            <joint type="free" />
        </body>

        <body name="goal_site" pos="0 0 0">
            <geom name="goal" type="sphere" size="{goal_thresh}" rgba="0 1 0 0.1"
             contype="0" conaffinity="0" />
        </body>

        <geom name="floor" pos="0 0 -0.9196429" size="15 15 0.01" type="plane" material="grid"/>
    </worldbody>

  <option>
  </option>
</mujoco>
"""


# -----------------------------
# ONNX helpers
# -----------------------------
def load_onnx_session(onnx_path: str):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    rgb_name = None
    state_name = None
    state_dim = None
    for i in inputs:
        n = i.name
        nlow = n.lower()
        if nlow == "rgb":
            rgb_name = i.name
        elif nlow == "state":
            state_name = i.name
            # shape should be [None, S]
            if isinstance(i.shape, list) and len(i.shape) == 2 and isinstance(i.shape[1], int):
                state_dim = i.shape[1]
            else:
                try:
                    state_dim = int(i.shape[1])
                except Exception:
                    state_dim = None
    if rgb_name is None:
        raise RuntimeError("ONNX model must have an input named 'rgb'")
    return sess, rgb_name, state_name, state_dim


# -----------------------------
# MuJoCo scene helpers
# -----------------------------
def setup_scene_and_renderer(camera_name: str, render_w: int, render_h: int):
    xml = build_scene_xml(camera_name)
    spec = mj.MjSpec.from_string(xml)

    # Remove the default light from the franka_panda model
    spec.delete(spec.light('franka_panda_top'))

    # Randomize cube pose
    cube_pos = np.zeros((3,))
    cube_pos[:2] = np.random.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=(2,))
    cube_pos[2] = cube_half_size
    cube_euler = np.zeros((3,))
    cube_euler[2] = np.random.uniform(-np.pi, np.pi)
    cube_quat = np.zeros((4,), dtype=np.float64)
    mj.mju_euler2Quat(cube_quat, cube_euler, "xyz")
    spec.geom('cube').pos = cube_pos
    spec.geom('cube').quat = cube_quat

    # Randomize goal
    goal_pos = np.zeros((3,))
    goal_pos[:2] = np.random.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=(2,))
    goal_pos[2] = np.random.uniform(0, max_goal_height) + cube_half_size
    spec.geom('goal').pos = goal_pos

    model = spec.compile()
    data = mj.MjData(model)

    # Initial joint configuration (arm + gripper if present)
    default_qpos = np.array([0.0, np.pi / 8, 0.0, -5 * np.pi / 8, 0.0, 3 * np.pi / 4, np.pi / 4, 0.04, 0.04], dtype=np.float64)
    nq_set = min(model.nq, default_qpos.shape[0])
    data.qpos[:nq_set] = default_qpos[:nq_set].copy()
    mj.mj_forward(model, data)

    renderer = mj.Renderer(model, render_w, render_h)
    return model, data, renderer


def get_camera_rgb(renderer: mj.Renderer, data: mj.MjData, camera_name: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()  # float [H,W,3] in [0,1]
    # BYD Github Copilot
    # the elements in img is actually of range [0,255]
    # there's no need to multiply it by 255 again!
    # As shown by the following line
    # print(img.sum() / np.array(img.shape).prod())
    img = img.astype(np.float32)
    # img = (img * 255.0).astype(np.float32)  # float32 [0,255], HWC
    return img


# -----------------------------
# Actuator and joint mapping
# -----------------------------
_joint_num_pat = re.compile(r'(?:franka_)?panda.*joint\s*([0-9]+)', re.IGNORECASE)
_arm_joint_hint = re.compile(r'(?:franka_)?panda.*joint', re.IGNORECASE)
_gripper_hint = re.compile(r'(finger|hand|grip)', re.IGNORECASE)

def name_or_empty(name_ptr: Optional[str]) -> str:
    return name_ptr if isinstance(name_ptr, str) else ""

def build_action_mapping(model: mj.MjModel) -> Dict[str, List[int]]:
    """
    Build lists of actuator indices for arm and gripper based on names and their target joints.
    Returns:
      mapping = {
        "arm_actuators": [actuator indices ordered by joint number 1..7 if possible],
        "gripper_actuators": [actuator indices for fingers]
      }
    """
    # # Gather joint number -> joint id
    # jointnum_to_jid: Dict[int, int] = {}
    # print(model.njnt)
    # for j in range(model.njnt):
    #     jname = name_or_empty(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j))
    #     print(jname)
    #     if _arm_joint_hint.search(jname):
    #         m = _joint_num_pat.search(jname)
    #         print(f"挖他都只{m}")
    #         if m:
    #             jointnum_to_jid[int(m.group(1))] = j

    # print(jointnum_to_jid)

    # # Map actuators to their target joints
    # arm_actuators: Dict[int, int] = {}  # jointnum -> actuator id
    # gripper_actuators: List[int] = []
    # print(model.njnt)
    # for a in range(model.nu):
    #     aname = name_or_empty(mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, a))
    #     # actuator target joint id
    #     jid = int(model.actuator_trnid[a, 0]) if model.actuator_trnid[a, 0] >= 0 else -1
    #     if jid >= 0:
    #         # Check if this target joint corresponds to numbered panda joints
    #         matched = False
    #         for jointnum, jj in jointnum_to_jid.items():
    #             if jj == jid:
    #                 arm_actuators[jointnum] = a
    #                 matched = True
    #                 break
    #         if not matched and _gripper_hint.search(aname):
    #             gripper_actuators.append(a)
    #     else:
    #         # No joint target; rely on name hints
    #         if _gripper_hint.search(aname):
    #             gripper_actuators.append(a)

    # # Order arm actuators by joint number
    # ordered_arms = [arm_actuators[k] for k in sorted(arm_actuators.keys())]

    # return {
    #     "arm_actuators": ordered_arms,
    #     "gripper_actuators": gripper_actuators,
    # }
    return {
        "arm_actuators": [0, 1, 2, 3, 4, 5, 6],
        "gripper_actuators": [7],
    }


def joint_ranges_for_actuators(model: mj.MjModel, act_inds: List[int]) -> List[Tuple[float, float]]:
    """
    For each actuator controlling a joint, return that joint's range.
    """
    ranges = []
    for a in act_inds:
        jid = int(model.actuator_trnid[a, 0]) if model.actuator_trnid[a, 0] >= 0 else -1
        if jid >= 0:
            lo, hi = model.jnt_range[jid]
            # If unbounded (0,0), set wide limits
            if lo == 0.0 and hi == 0.0:
                lo, hi = -1e3, 1e3
        else:
            lo, hi = -1e3, 1e3
        ranges.append((float(lo), float(hi)))
    return ranges


def actuator_ctrlrange(model: mj.MjModel, a: int) -> Tuple[float, float]:
    cr = model.actuator_ctrlrange[a]
    lo, hi = float(cr[0]), float(cr[1])
    if lo == 0.0 and hi == 0.0:
        lo, hi = -1e3, 1e3
    return lo, hi


# -----------------------------
# State feature extractor
# -----------------------------
def find_body_id(model: mj.MjModel, candidates: List[str]) -> Optional[int]:
    for name in candidates:
        try:
            return mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        except Exception:
            continue
    return None


def find_ee_body_id(model: mj.MjModel) -> Optional[int]:
    # Try common Panda EE body/site names
    body_candidates = [
        "franka_panda_link7", "franka_panda_hand", "panda_link7", "panda_hand", "hand", "tcp", "ee"
    ]
    bid = find_body_id(model, body_candidates)
    if bid is not None:
        return bid
    # Fallback: find the last body whose name contains 'panda' and 'link'
    last_match = None
    for b in range(model.nbody):
        name = name_or_empty(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b)).lower()
        if "panda" in name and "link" in name:
            last_match = b
    return last_match


def quat_to_vec(q: np.ndarray) -> np.ndarray:
    # Ensure wxyz or xyzw? Mujoco uses (w,x,y,z) in data.xquat
    return q.astype(np.float32).copy()

def is_cube_been_grasped(model: mj.MjModel, data: mj.MjData, finger_names: List[str], cube_name: str, force_threshold: float = 1e-6) -> bool:
    """
    Checks if the Franka Panda fingers are grasping the cube by checking contact forces.
    
    Parameters:
    - model: MuJoCo model
    - data: MuJoCo data object
    - finger_names: List of body names for the fingers (e.g., ['franka_panda_left_finger', 'franka_panda_right_finger'])
    - cube_name: The name of the cube body (e.g., 'cube')
    - force_threshold: Minimum contact force considered as "grasping"

    Returns:
    - bool: True if any of the fingers are grasping the cube, False otherwise
    """

    for i in range(data.ncon):
        contact = data.contact[i]
        body1_name = name_or_empty(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, contact.geom1))
        body2_name = name_or_empty(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, contact.geom2))
        if (body1_name in finger_names and body2_name == cube_name) or (body2_name in finger_names and body1_name == cube_name):
            force_magnitude = np.linalg.norm(contact.force)
            if force_magnitude > force_threshold:
                return True

    return False

def compute_state_vector(model: mj.MjModel,
                          data: mj.MjData,
                          arm_actuators: List[int],
                          gripper_actuators: List[int],
                          target_by_act: np.ndarray,
                          state_dim: Optional[int]) -> np.ndarray:
    """
    Compose a heuristic state vector approximating ManiSkill's 'state':
    - Joint qpos (9 if available)
    - Joint qvel (9 if available)
    - Whether the cube have been grasped (a boolean)
    - EE pose: position(3) + quaternion(4)
    - Goal position(3)
    Then pad or truncate to match ONNX's expected state_dim.

    The order of elements in the ManiSkill state vector
    can be obtained in the following way:

    env = gym.make("PickCube-v1", obs_mode="rgb")
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env.base_env._get_obs_agent()
    env.base_env._get_obs_extra({"is_grasped": True})
    """

    vec = []

    arm_qpos = []
    arm_qvel = []

    # Joint qpos and qvel
    # print(model.njnt)
    for jid in range(9):
        qadr = model.jnt_qposadr[jid]
        vadr = model.jnt_dofadr[jid] if model.jnt_dofadr[jid] >= 0 else None
        arm_qpos.append(float(data.qpos[qadr]))
        arm_qvel.append(float(data.qvel[vadr]) if vadr is not None else 0.0)
    vec.extend(arm_qpos)
    vec.extend(arm_qvel)

    # Whether the cube has been grasped
    # it is equivalent to checking whether the two Franka Panda fingers
    # are in contact with the cube
    is_grasped = is_cube_been_grasped(model, data, 'franka_panda_left_finger', 'cube', 0)
    is_grasped &= is_cube_been_grasped(model, data, 'franka_panda_right_finger', 'cube', 0)
    vec.append(1.0 if is_grasped else 0.0)

    # EE pose
    ee_bid = find_ee_body_id(model)
    if ee_bid is not None:
        ee_pos = data.xpos[ee_bid].astype(np.float32)  # (3,)
        ee_quat = data.xquat[ee_bid].astype(np.float32)  # (4,) (w,x,y,z)
        vec.extend(ee_pos.tolist())
        vec.extend(ee_quat.tolist())
        ee_pos3 = ee_pos
    else:
        vec.extend([0.0] * 7)
        ee_pos3 = np.zeros(3, dtype=np.float32)

    # Goal position (body 'goal_site' or geom 'goal' parent)
    # try:
        # goal_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "goal_site")
        # goal_pos = data.xpos[goal_bid].astype(np.float32)
    # except Exception:
    #     # Try geom
    #     try:
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "goal")
    goal_pos = data.geom_xpos[gid].astype(np.float32)
    #     except Exception:
    #         goal_pos = np.zeros(3, dtype=np.float32)
    vec.extend(goal_pos.tolist())

    # print(len(vec))
    assert len(vec) == 29

    vec_np = np.array(vec, dtype=np.float32)

    if state_dim is None:
        # No expected dim; return as-is
        return vec_np
    # Pad or truncate to expected length
    if vec_np.shape[0] < state_dim:
        pad = np.zeros((state_dim - vec_np.shape[0],), dtype=np.float32)
        vec_np = np.concatenate([vec_np, pad], axis=0)
    elif vec_np.shape[0] > state_dim:
        vec_np = vec_np[:state_dim]
    print(vec_np)
    return vec_np


# -----------------------------
# Control application
# -----------------------------
def apply_action_deltas(model: mj.MjModel,
                        data: mj.MjData,
                        action: np.ndarray,
                        arm_actuators: List[int],
                        gripper_actuators: List[int],
                        target_by_act: np.ndarray,
                        action_scale: float):
    """
    Map action dims to actuators:
      - First len(arm_actuators) dims -> arm position target deltas
      - Next dims -> gripper targets (open/close) as deltas
    target_by_act holds per-actuator target values that we update and write into data.ctrl.
    """
    a = np.asarray(action, dtype=np.float32)
    # Split
    na = len(arm_actuators)
    ng = len(gripper_actuators)
    arm_cmd = a[:na] if a.shape[0] >= na else np.zeros(na, dtype=np.float32)
    grip_cmd = a[na:na + ng] if a.shape[0] >= na + ng else np.zeros(ng, dtype=np.float32)

    # print(arm_cmd)
    # print(grip_cmd)
    # print(target_by_act)

    # The actuators should be updated
    # as if target_by_act had been normalized from [lo, hi] to [-1, 1]
    # See https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html#deep-dive-example-of-the-franka-emika-panda-robot-controllers

    # Update arm actuator targets with clamping to joint ranges
    arm_ranges = joint_ranges_for_actuators(model, arm_actuators)
    for i, act_id in enumerate(arm_actuators):
        lo, hi = actuator_ctrlrange(model, act_id)
        # Prefer joint range if actuator ctrlrange is narrow/unknown
        jlo, jhi = arm_ranges[i]
        # Update target
        tgt = target_by_act[act_id] + action_scale * (hi - lo) * float(arm_cmd[i])
        # Clamp by joint range first, then actuator ctrlrange
        tgt = np.clip(tgt, jlo, jhi)
        tgt = np.clip(tgt, lo, hi)
        target_by_act[act_id] = tgt
        data.ctrl[act_id] = tgt

    # Update gripper targets (often position actuators with small ranges)
    for i, act_id in enumerate(gripper_actuators):
        lo, hi = actuator_ctrlrange(model, act_id)
        tgt = target_by_act[act_id] + action_scale * (hi - lo) * float(grip_cmd[i])
        tgt = np.clip(tgt, lo, hi)
        target_by_act[act_id] = tgt
        data.ctrl[act_id] = tgt


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, default=os.environ.get("POLICY_ONNX_PATH", "model_pickcube_rgb.onnx"),
                   help="Path to exported ONNX policy.")
    p.add_argument("--policy", type=str, choices=["deterministic", "stochastic"], default="deterministic",
                   help="Use deterministic mean action or add Gaussian exploration.")
    p.add_argument("--stochastic-std", type=float, default=0.01,
                   help="Std dev for Gaussian sampling around the ONNX mean (used if policy=stochastic).")
    p.add_argument("--action-scale", type=float, default=DEFAULT_ACTION_SCALE,
                   help="Scale applied to action deltas before sending to actuators.")
    p.add_argument("--ctrl-hz", type=float, default=DEFAULT_CTRL_HZ,
                   help="Control loop frequency in Hz.")
    p.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    p.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH)
    p.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT)
    p.add_argument("--print-mapping", action="store_true", help="Print resolved actuator mapping.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load ONNX model
    sess, rgb_name, state_name, state_dim = load_onnx_session(args.onnx)
    print(state_dim)
    expects_state = state_name is not None

    # Create scene
    model, data, renderer = setup_scene_and_renderer(args.camera_name, args.render_width, args.render_height)

    # Resolve actuator mapping
    mapping = build_action_mapping(model)
    # print(mapping)
    arm_actuators = mapping["arm_actuators"]
    gripper_actuators = mapping["gripper_actuators"]

    if args.print_mapping:
        print(f"Arm actuators (ordered): {[(i, name_or_empty(mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i))) for i in arm_actuators]}")
        print(f"Gripper actuators: {[(i, name_or_empty(mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i))) for i in gripper_actuators]}")

    # Initialize actuator targets to current qpos when possible, else zeros
    target_by_act = np.zeros(model.nu, dtype=np.float32)
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0]) if model.actuator_trnid[a, 0] >= 0 else -1
        if jid >= 0:
            qadr = model.jnt_qposadr[jid]
            target_by_act[a] = float(data.qpos[qadr])
        else:
            # default within ctrlrange midpoint
            lo, hi = actuator_ctrlrange(model, a)
            target_by_act[a] = 0.5 * (lo + hi)
        data.ctrl[a] = target_by_act[a]

    # Viewer loop

    rgb = get_camera_rgb(renderer, data, args.camera_name)
    from PIL import Image
    image = Image.fromarray(rgb.astype(np.uint8))
    image.save("mujoco.png")

    with viewer.launch_passive(model, data) as v:
        t_last = time.time()
        dt_target = 1.0 / max(args.ctrl_hz, 1e-6)

        while v.is_running():
            now = time.time()
            if now - t_last >= dt_target:
                # 1) Render RGB
                rgb = get_camera_rgb(renderer, data, args.camera_name)   # [H,W,3], float32 [0,255]

                # # export rgb to image
                # from PIL import Image
                # image = Image.fromarray(rgb.astype(np.uint8))
                # image.save("output.png")

                rgb_batch = np.expand_dims(rgb, axis=0)                  # [1,H,W,3]

                # 2) Build state input
                ort_inputs = {rgb_name: rgb_batch}
                if expects_state:
                    state_vec = compute_state_vector(model, data, arm_actuators, gripper_actuators, target_by_act, state_dim)
                    # print(state_vec)
                    # state_vec = np.zeros_like(state_vec)
                    # state_dim
                    ort_inputs[state_name] = state_vec.reshape(1, -1).astype(np.float32)

                # 3) Policy inference
                output = sess.run(None, ort_inputs)  # action_mean: [1, A]
                # print(output)
                action_mean, _value = output
                mean = action_mean[0].astype(np.float32)

                if args.policy == "stochastic":
                    std = float(args.stochastic_std)
                    action = mean + np.random.randn(*mean.shape).astype(np.float32) * std
                else:
                    action = mean

                # print(action)

                # 4) Apply action via actuator mapping
                apply_action_deltas(model, data, action, arm_actuators, gripper_actuators, target_by_act, args.action_scale)

                t_last = now

            # Step simulation faster than control rate
            mj.mj_step(model, data)
            v.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()