#!/usr/bin/env python3
"""RL environment for OpenArm target-conditioned pick (DirectRLEnv).

Why RL: the scripted IK oracle (``scripts/oracle_pick_ik.py``) grasps only
~40-83% because position-only IK lets the wrist drift to a position-dependent
orientation, and a single captured orientation does not generalize (see
``synthetic_smolvla/README.md`` -> "Optimization attempts"). Instead of hand
-tuning the wrist, this env lets PPO *learn* a closed-loop joint policy that
reaches, orients, closes, and lifts the requested object, rewarded by the
object's measured height rise in physics.

Task: the four objects (orange ball, red/green/blue cubes) sit in the right
arm's reachable pocket. Each episode a target object is chosen at random; the
target identity is given to the policy as a one-hot in the observation, so the
trained policy is language/target conditioned and can serve as the SmolVLA
demonstration oracle. Only the right arm (7 joints) + gripper are actuated; the
left arm is held at its rest pose. Joint limits use Isaac's soft limits; the
project's safe-degree contract is enforced when this policy is later replayed as
the oracle (the env clamps to Isaac soft limits, which are within the USD).

This is simulation-only. Nothing here opens a CAN bus or moves a real robot.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG


# Object layout inside the right arm's confirmed reachable pocket for a base at
# (0,0,0.40): x=0.30, y in [-0.26,-0.08], z~0.55 (see reports/openarm_reach_probe.json).
OBJECTS = [
    {"name": "orange_ball", "shape": "sphere", "color": (1.0, 0.45, 0.02), "radius": 0.020, "pos": (0.30, -0.26, 0.55)},
    {"name": "red_cube", "shape": "cube", "color": (0.9, 0.03, 0.03), "size": (0.035, 0.035, 0.035), "pos": (0.30, -0.20, 0.55)},
    {"name": "green_cube", "shape": "cube", "color": (0.05, 0.70, 0.12), "size": (0.035, 0.035, 0.035), "pos": (0.30, -0.14, 0.55)},
    {"name": "blue_cube", "shape": "cube", "color": (0.04, 0.20, 0.95), "size": (0.035, 0.035, 0.035), "pos": (0.30, -0.08, 0.55)},
]
NUM_OBJECTS = len(OBJECTS)

# Right-arm rest pose (deg) matching configs/scene_openarm_four_objects.yaml.
RIGHT_RESET_DEG = {1: 0.0, 2: 20.0, 3: 0.0, 4: 55.0, 5: 0.0, 6: 15.0, 7: 0.0}
LEFT_RESET_DEG = {1: 0.0, 2: -20.0, 3: 0.0, 4: 55.0, 5: 0.0, 6: -15.0, 7: 0.0}
FINGER_OPEN_M = 0.044  # gripper fully open
TABLE_TOP_Z = 0.53
SUCCESS_RISE_M = 0.08  # target height rise that counts as a successful lift


def _object_spawn(obj: dict) -> RigidObjectCfg:
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=obj["color"], roughness=0.55)
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=5.0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03 if obj["shape"] == "sphere" else 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9, restitution=0.0),
        visual_material=material,
    )
    if obj["shape"] == "sphere":
        spawn = sim_utils.SphereCfg(radius=obj["radius"], **common)
    else:
        spawn = sim_utils.CuboidCfg(size=obj["size"], **common)
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{obj['name']}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=obj["pos"], rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class OpenArmPickEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 3
    action_space = 8   # 7 right-arm joint deltas + 1 gripper command
    observation_space = 34  # see _get_observations
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot: bimanual OpenArm, base raised to table height, right arm active.
    robot: ArticulationCfg = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # control
    action_scale = 0.6           # rad/s-ish scale on arm joint deltas
    dt_action_scale = 1.0

    # reward scales
    reach_reward_scale = 1.0
    near_bonus = 0.5
    straddle_reward_scale = 2.0   # object centred between the two fingers
    held_bonus = 4.0              # fingers closed AND object centred (a grasp)
    held_lift_scale = 60.0        # height gained WHILE held (the key signal)
    lift_reward_scale = 20.0
    success_bonus = 10.0
    wrong_penalty_scale = 20.0
    action_penalty_scale = 0.01
    grip_reward_scale = 0.4

    def __post_init__(self):
        # Mount the base at table height and start both arms at their rest pose
        # (right arm active, left arm folded, grippers open).
        self.robot.init_state.pos = (0.0, 0.0, 0.40)
        self.robot.init_state.joint_pos = make_default_joint_pos()


class OpenArmPickEnv(DirectRLEnv):
    """PPO env: pick the requested object with the OpenArm right arm."""

    cfg: OpenArmPickEnvCfg

    def __init__(self, cfg: OpenArmPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Joint / body indices for the active (right) arm + gripper.
        self.right_arm_ids, _ = self._robot.find_joints([f"openarm_right_joint{i}" for i in range(1, 8)], preserve_order=True)
        self.finger_ids, _ = self._robot.find_joints("openarm_right_finger_joint.*")
        self.right_arm_ids = torch.tensor(self.right_arm_ids, device=self.device, dtype=torch.long)
        self.finger_ids = torch.tensor(self.finger_ids, device=self.device, dtype=torch.long)
        self.tcp_idx = self._robot.find_bodies("openarm_right_ee_tcp")[0][0]
        self.lfinger_idx = self._robot.find_bodies("openarm_right_left_finger")[0][0]
        self.rfinger_idx = self._robot.find_bodies("openarm_right_right_finger")[0][0]

        # Soft joint limits for the right arm (for clamping integrated targets).
        lo = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        hi = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self.arm_lo = lo[self.right_arm_ids]
        self.arm_hi = hi[self.right_arm_ids]

        # Per-env running targets and bookkeeping.
        self.arm_targets = torch.zeros((self.num_envs, 7), device=self.device)
        self.target_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_init_z = torch.zeros(self.num_envs, device=self.device)
        self.obj_init_z = torch.zeros((self.num_envs, NUM_OBJECTS), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Rigid objects.
        self._objects: list[RigidObject] = []
        for obj in OBJECTS:
            ro = RigidObject(_object_spawn(obj))
            self.scene.rigid_objects[obj["name"]] = ro
            self._objects.append(ro)

        # A static table top so objects rest at the reachable height.
        table_cfg = sim_utils.CuboidCfg(
            size=(0.40, 0.40, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.50, 0.44), roughness=0.7),
        )
        table_cfg.func("/World/envs/env_0/Table", table_cfg, translation=(0.32, -0.16, TABLE_TOP_Z - 0.02))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78))
        light_cfg.func("/World/Light", light_cfg)

    # ---- helpers ---------------------------------------------------------

    def _obj_pos_local(self) -> torch.Tensor:
        """(num_envs, NUM_OBJECTS, 3) object positions in env-local frame."""
        out = torch.zeros((self.num_envs, NUM_OBJECTS, 3), device=self.device)
        for i, ro in enumerate(self._objects):
            out[:, i, :] = ro.data.root_pos_w - self.scene.env_origins
        return out

    def _tcp_pos_local(self) -> torch.Tensor:
        return self._robot.data.body_pos_w[:, self.tcp_idx] - self.scene.env_origins

    def _gather_target(self, obj_local: torch.Tensor) -> torch.Tensor:
        idx = self.target_idx.view(-1, 1, 1).expand(-1, 1, 3)
        return obj_local.gather(1, idx).squeeze(1)

    # ---- RL API ----------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Integrate arm joint deltas, clamp to soft limits.
        self.arm_targets = torch.clamp(
            self.arm_targets + self.actions[:, :7] * self.cfg.action_scale * self.dt * self.cfg.dt_action_scale,
            self.arm_lo, self.arm_hi,
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.arm_targets, joint_ids=self.right_arm_ids.tolist())
        # Gripper: action[-1] in [-1,1] -> finger opening [0, FINGER_OPEN_M].
        finger = ((self.actions[:, 7:8] + 1.0) * 0.5 * FINGER_OPEN_M).expand(-1, len(self.finger_ids))
        self._robot.set_joint_position_target(finger, joint_ids=self.finger_ids.tolist())

    def _get_observations(self) -> dict:
        arm_pos = self._robot.data.joint_pos[:, self.right_arm_ids]
        arm_pos_scaled = 2.0 * (arm_pos - self.arm_lo) / (self.arm_hi - self.arm_lo) - 1.0
        arm_vel = self._robot.data.joint_vel[:, self.right_arm_ids] * 0.1
        grip = self._robot.data.joint_pos[:, self.finger_ids].sum(dim=-1, keepdim=True)

        tcp = self._tcp_pos_local()
        obj_local = self._obj_pos_local()
        rel = (obj_local - tcp.unsqueeze(1)).reshape(self.num_envs, NUM_OBJECTS * 3)
        onehot = torch.zeros((self.num_envs, NUM_OBJECTS), device=self.device)
        onehot.scatter_(1, self.target_idx.view(-1, 1), 1.0)

        obs = torch.cat([arm_pos_scaled, arm_vel, grip, tcp, rel, onehot], dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        obj_local = self._obj_pos_local()
        tcp = self._tcp_pos_local()
        target_pos = self._gather_target(obj_local)

        d = torch.norm(tcp - target_pos, p=2, dim=-1)
        reach = 1.0 / (1.0 + d**2)
        reach = reach * reach
        near = (d < 0.04).float() * self.cfg.near_bonus

        # Fingertip geometry: object centred between the two fingers + gripper closed
        # = a grasp. This dense signal lets the policy LEARN to grasp instead of
        # having to randomly stumble into a successful lift.
        lf = self._robot.data.body_pos_w[:, self.lfinger_idx] - self.scene.env_origins
        rf = self._robot.data.body_pos_w[:, self.rfinger_idx] - self.scene.env_origins
        finger_mid = 0.5 * (lf + rf)
        straddle_d = torch.norm(target_pos - finger_mid, p=2, dim=-1)
        straddle = (1.0 / (1.0 + (10.0 * straddle_d) ** 2)) * self.cfg.straddle_reward_scale

        grip = self._robot.data.joint_pos[:, self.finger_ids].sum(dim=-1)
        grip_full = FINGER_OPEN_M * len(self.finger_ids)
        grip_closed = (grip_full - grip) / grip_full  # 0 open -> 1 closed
        grip_reward = (d < 0.05).float() * grip_closed * self.cfg.grip_reward_scale

        # "Held": object centred between fingers AND gripper mostly closed.
        held = ((straddle_d < 0.035) & (grip_closed > 0.5)).float()
        held_reward = held * self.cfg.held_bonus

        obj_z = obj_local[:, :, 2]
        rises = obj_z - self.obj_init_z
        target_rise = rises.gather(1, self.target_idx.view(-1, 1)).squeeze(1)
        lift = torch.clamp(target_rise, min=0.0) * self.cfg.lift_reward_scale
        held_lift = held * torch.clamp(target_rise, min=0.0) * self.cfg.held_lift_scale

        # Wrong-object penalty: any non-target rising.
        wrong_mask = torch.ones_like(rises)
        wrong_mask.scatter_(1, self.target_idx.view(-1, 1), 0.0)
        wrong_rise = torch.clamp(rises * wrong_mask, min=0.0).sum(dim=-1)
        wrong_pen = wrong_rise * self.cfg.wrong_penalty_scale

        action_pen = torch.sum(self.actions**2, dim=-1) * self.cfg.action_penalty_scale
        success = (target_rise > SUCCESS_RISE_M).float() * self.cfg.success_bonus

        rewards = (
            self.cfg.reach_reward_scale * reach + near + straddle + grip_reward
            + held_reward + lift + held_lift + success - wrong_pen - action_pen
        )
        self.extras["log"] = {
            "reach": (self.cfg.reach_reward_scale * reach).mean(),
            "straddle": straddle.mean(),
            "held_rate": held.mean(),
            "held_lift": held_lift.mean(),
            "lift": lift.mean(),
            "wrong_penalty": (-wrong_pen).mean(),
            "success_rate": (target_rise > SUCCESS_RISE_M).float().mean(),
            "mean_target_rise_m": target_rise.mean(),
        }
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_local = self._obj_pos_local()
        target_z = self._gather_target(obj_local)[:, 2]
        target_rise = target_z - self.target_init_z
        lifted = target_rise > SUCCESS_RISE_M
        fell = target_z < (TABLE_TOP_Z - 0.15)  # knocked off the table
        terminated = lifted | fell
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Robot back to rest pose (+ small arm noise), gripper open.
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        noise = sample_uniform(-0.05, 0.05, (n, 7), self.device)
        joint_pos[:, self.right_arm_ids] = torch.clamp(
            joint_pos[:, self.right_arm_ids] + noise, self.arm_lo, self.arm_hi
        )
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.arm_targets[env_ids] = joint_pos[:, self.right_arm_ids]

        # Objects back to their spawn poses (+ small xy jitter inside the pocket).
        origins = self.scene.env_origins[env_ids]
        for i, ro in enumerate(self._objects):
            root = ro.data.default_root_state[env_ids].clone()
            base = torch.tensor(OBJECTS[i]["pos"], device=self.device)
            jitter = torch.zeros((n, 3), device=self.device)
            jitter[:, 0] = sample_uniform(-0.015, 0.015, (n,), self.device)
            jitter[:, 1] = sample_uniform(-0.008, 0.008, (n,), self.device)
            root[:, 0:3] = base.unsqueeze(0) + jitter + origins
            root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            root[:, 7:] = 0.0
            ro.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
            ro.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

        # New random target per reset env.
        self.target_idx[env_ids] = torch.randint(0, NUM_OBJECTS, (n,), device=self.device)

        # Record object baseline heights for rise measurement.
        obj_local = self._obj_pos_local()
        self.obj_init_z[env_ids] = obj_local[env_ids, :, 2]
        self.target_init_z[env_ids] = obj_local[env_ids, :, 2].gather(1, self.target_idx[env_ids].view(-1, 1)).squeeze(1)


def make_default_joint_pos() -> dict[str, float]:
    """Right/left rest pose + open gripper for the OpenArm init state."""
    jp: dict[str, float] = {}
    for i in range(1, 8):
        jp[f"openarm_right_joint{i}"] = math.radians(RIGHT_RESET_DEG[i])
        jp[f"openarm_left_joint{i}"] = math.radians(LEFT_RESET_DEG[i])
    jp["openarm_right_finger_joint.*"] = FINGER_OPEN_M
    jp["openarm_left_finger_joint.*"] = FINGER_OPEN_M
    return jp
