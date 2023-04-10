import habitat_sim
from habitat_sim.logging import logger
from settings import default_sim_settings, make_cfg

import habitat_sim.physics as phy
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum
from scipy.spatial.transform import Rotation as R

import cv2
import magnum as mn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
import pathlib
import json

from fairmotion.core import motion, velocity
from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.ops.motion import cut, rotate, translate
import sys
sys.path.append('/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data')
import os
import imageio

#### Constants ###
ROOT, LAST = 0, -1

def global_correction_quat(up_v: mn.Vector3, forward_v: mn.Vector3) -> mn.Quaternion:
    """
    Given the upward direction and the forward direction of a local space frame, this methd produces
    the correction quaternion to convert the frame to global space (+Y up, -Z forward).
    """
    if up_v.normalized() != mn.Vector3.y_axis():
        angle1 = mn.math.angle(up_v.normalized(), mn.Vector3.y_axis())
        axis1 = mn.math.cross(up_v.normalized(), mn.Vector3.y_axis())
        rotation1 = mn.Quaternion.rotation(angle1, axis1)
        forward_v = rotation1.transform_vector(forward_v)
    else:
        rotation1 = mn.Quaternion()

    forward_v = forward_v * (mn.Vector3(1.0, 1.0, 1.0) - mn.Vector3.y_axis())
    angle2 = mn.math.angle(forward_v.normalized(), -1 * mn.Vector3.z_axis())
    axis2 = mn.Vector3.y_axis()
    rotation2 = mn.Quaternion.rotation(angle2, axis2)

    return rotation2 * rotation1

class WalkerEnvHabitat():
    def __init__(
            self, 
            sim_settings, 
            bvh_path, 
            rotation_offset=mn.Quaternion(), 
            translate_offset=mn.Vector3([0, 0, 0]), 
            motion_start=None, 
            motion_end=None,
            motion_stepper=0,
            fps=200, 
            frameskip=4,
            urdf_path="CIRCLE_assets/subjects/1_agent.urdf", #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/quadruped/quadruped.urdf" #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/humanoid/nao.urdf" #"CIRCLE_assets/subjects/1_agent.urdf"
            ref_urdf_path="CIRCLE_assets/subjects/1.urdf" #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/quadruped/quadruped.urdf" #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/humanoid/nao.urdf" #"CIRCLE_assets/subjects/1_agent.urdf"
    ):
        self.sim_settings = sim_settings
        self.urdf_path = urdf_path
        self.ref_urdf_path = ref_urdf_path
        self.motion_type = phy.MotionType.DYNAMIC

        self.rotation_offset: Optional[mn.Quaternion] = rotation_offset
        self.translate_offset: Optional[mn.Vector3] = translate_offset
        self.bvh_path = bvh_path
        self.motion_stepper = motion_stepper
        self.motion_start: int = motion_start
        self.motion_end: int = motion_end

        self.fps = fps
        self.frameskip = frameskip

        self.final_rotation_correction = global_correction_quat(mn.Vector3.z_axis(), mn.Vector3.x_axis())
        self._reconfigure_sim()
        self._load_motion()

        self.art_obj_mgr = self.sim.get_articulated_object_manager()
        self._add_stage()
        self._add_agent_model()
        self._add_ref_model()
    
    def _reconfigure_sim(self):
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        if self.sim_settings["stage_requires_lighting"]:
            logger.info("Setting synthetic lighting override for stage.")
            self.cfg.sim_cfg.override_scene_light_defaults = True
            self.cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        self.sim = habitat_sim.Simulator(self.cfg)

        # post reconfigure
        self.active_scene_graph = self.sim.get_active_scene_graph()
       
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.agent_body_node = self.default_agent.scene_node
       
        self.render_camera = self.agent_body_node.node_sensor_suite.get("color_sensor")
        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name

    def _load_motion(self) -> None:
        """
        Loads the motion
        """
        # loading text because the setup pauses here during motion load
        logger.info("Loading...")
        self.motion = bvh.load(file=self.bvh_path) #, np.array(self.final_rotation_correction.to_matrix())) #cut(bvh.load(file=self.bvh_path), self.motion_start, self.motion_end)
        # self.motion = translate(self.motion, np.array([0,2,0]))
        logger.info("Done Loading.")

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5

        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config
    
    def _add_stage(self):
        self.stage = self.art_obj_mgr.add_articulated_object_from_urdf(
            filepath="simpleplane.urdf"
        )
        assert self.stage.is_alive
        self.stage.motion_type = self.motion_type
        self.stage.rotation = self.final_rotation_correction
        self.stage.translation = mn.Vector3(0, 0, 0)

    def _add_agent_model(self):
        # self._hide_existing_model()

        self.agent_model = self.art_obj_mgr.add_articulated_object_from_urdf(
            filepath=self.urdf_path,
        )

        assert self.agent_model.is_alive

        # change motion_type to KINEMATIC
        self.agent_model.motion_type = self.motion_type

    def _add_ref_model(self):

        self.ref_model = self.art_obj_mgr.add_articulated_object_from_urdf(
            filepath=self.ref_urdf_path,
        )

        assert self.ref_model.is_alive

        # change motion_type to KINEMATIC
        self.ref_model.motion_type = phy.MotionType.KINEMATIC

    def _hide_existing_model(self):
        if self.agent_model:
            self.art_obj_mgr.remove_object_by_handle(self.agent_model.handle)
            self.model = None

    def step(self, action):
        is_done = False
        # self.motion_stepper = (self.motion_stepper + 1) % len(self.motion.poses)
        # print(f"action={action}")
        # self._next_ref_pose()
        posbefore = np.array(self.agent_model.translation)[2]

        for _ in range(self.frameskip):
            # self.agent_model.add_joint_forces(action / 1000)
            self.sim.step_world(1.0 / self.fps)

        # print(f"self.sim.data={self.sim.data}, qpos={self.sim.data.qpos}, qpos_shape={self.sim.data.qpos.shape}")
        # self.do_simulation_with_pd(a, self.frame_skip)
        
        height, posafter = np.array(self.agent_model.translation)[1:3]

        alive_bonus = 1.0
        reward = (posafter - posbefore) * self.fps
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        terminated = False
        # terminated = not (height > self.init_height / 4 and height < 5 * self.init_height)
        if height > 50 * self.init_height or np.max(self.agent_model.joint_velocities) > 5:
            print(f"height={height}, init_height={self.init_height}, reward={reward}, terminated={terminated}, velocity={self.agent_model.joint_velocities}")

        return self.get_observation(), reward, terminated, False, {}
        
    def _next_ref_pose(self):
        new_pose, new_pose_T, new_root_translate, new_root_rotation = self.get_ref_pose()
        # self.ref_model.joint_positions = new_pose
        # self.ref_model.joint_velocities = self.get_ref_velocity()
        self.ref_model.rotation = new_root_rotation
        self.ref_model.translation = new_root_translate + mn.Vector3(0.5, 0, 0.5)

    def get_observation(self):
        return np.concatenate([self.agent_model.joint_positions, self.agent_model.joint_velocities]).ravel()
    
    def _calc_imitation_reward(self):
        return None
    
    def reset(self):
        # self.motion_stepper = np.random.randint(0, len(self.motion.poses))
        new_pose, new_pose_T, new_root_translate, new_root_rotation = self.get_ref_pose(raw=False)
        # self.agent_model.joint_positions = np.random.uniform(low=-0.0005, high=0.0005, size=len(self.agent_model.joint_positions))
        # self.agent_model.joint_velocities = np.random.uniform(low=-0.005, high=0.005, size=len(self.agent_model.joint_velocities))
        self.agent_model.rotation = global_correction_quat(mn.Vector3.z_axis(), mn.Vector3.x_axis())
        self.init_height = 1.2
        self.agent_model.translation = mn.Vector3(0, self.init_height, 0)
        self.origin = self.agent_model.translation

        new_pose, new_pose_T, new_root_translate, new_root_rotation = self.get_ref_pose()
        self.ref_model.joint_positions = new_pose
        self.ref_model.joint_velocities = self.get_ref_velocity()
        self.ref_model.rotation = new_root_rotation
        self.ref_model.translation = new_root_translate

        self.agent_body_node.translation = self.agent_model.translation + mn.Vector3(1, 0.8, 1)
        camera_position = self.agent_body_node.translation
        camera_look_at = self.agent_model.translation
        self.agent_body_node.rotation = mn.Quaternion.from_matrix(
            mn.Matrix4.look_at(
                camera_position, camera_look_at, np.array([0, 1.0, 0])  # up
            ).rotation()
        )

        return self.get_observation(), {}

    def get_ref_pose(self, raw=True):
        """
        Returns the current pose of the model
        """
        pose_Q, pose_T, tran, rot = self.convert_CMUamass_single_pose(self.motion.poses[self.motion_stepper], self.ref_model, raw=raw)
        return pose_Q, pose_T, tran, rot

    def get_ref_velocity(self):
        """
        Returns the current velocity of the model
        """
        pose1_Q, pose1_T, tran1, rot1 = self.convert_CMUamass_single_pose(self.motion.poses[self.motion_stepper], self.ref_model, raw=True)
        pose2_Q, pose2_T, tran2, rot2 = self.convert_CMUamass_single_pose(self.motion.poses[(self.motion_stepper+1)%self.motion.num_frames()], self.ref_model, raw=True)
        pose1 = motion.Pose(self.motion.poses[0].skel, pose1_T)
        pose2 = motion.Pose(self.motion.poses[0].skel, pose2_T)
        vel = velocity.Velocity(pose1, pose2, dt=self.motion.fps_inv)
        return vel.data_local[1:, :3].ravel()
    
    def convert_CMUamass_single_pose(
        self, pose, model, raw=False
    ) -> Tuple[List[float], mn.Vector3, mn.Quaternion]:
        """
        This conversion is specific to the datasets from CMU
        """
        new_pose = []
        new_pose_T = []

        # Root joint
        root_T = pose.get_transform(ROOT, local=False)
        new_pose_T.append(root_T)
        
        final_rotation_correction = mn.Quaternion()

        if not raw:
            final_rotation_correction = (
                global_correction_quat(mn.Vector3.y_axis(), mn.Vector3.x_axis())
                * self.rotation_offset
            )

        root_rotation = final_rotation_correction * mn.Quaternion.from_matrix(
            mn.Matrix3x3(root_T[0:3, 0:3])
        )
        root_translation = (
            self.translate_offset
            + final_rotation_correction.transform_vector(root_T[0:3, 3])
        )

        Q, _ = conversions.T2Qp(root_T)

        # Other joints
        for model_link_id in model.get_link_ids():
            joint_type = model.get_link_joint_type(model_link_id)

            if joint_type == phy.JointType.Fixed:
                continue

            joint_name = model.get_link_name(model_link_id)
            pose_joint_index = pose.skel.index_joint[joint_name]

            # When the target joint do not have dof, we simply ignore it

            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            if pose_joint_index is None:
                raise KeyError(
                    "Error: pose data does not have a transform for that joint name"
                )
            elif joint_type not in [phy.JointType.Spherical]:
                raise NotImplementedError(
                    f"Error: {joint_type} is not a supported joint type"
                )
            else:
                T = pose.get_transform(pose_joint_index, local=True)
                if joint_type == phy.JointType.Spherical:
                    Q, _ = conversions.T2Qp(T)

            new_pose += list(Q)
            new_pose_T.append(T)

        return new_pose, new_pose_T, root_translation, root_rotation    

    def draw_axes(self, axis_len=1.0):
        lr = self.sim.get_debug_line_render()
        # define some constants and globals the first time we run:
        opacity = 1.0
        red = mn.Color4(1.0, 0.0, 0.0, opacity)
        green = mn.Color4(0.0, 1.0, 0.0, opacity)
        blue = mn.Color4(0.0, 0.0, 1.0, opacity)
        # draw axes with x+ = red, y+ = green, z+ = blue
        lr.draw_transformed_line(self.origin, self.origin+mn.Vector3(axis_len, 0, 0), red)
        lr.draw_transformed_line(self.origin, self.origin+mn.Vector3(0, axis_len, 0), green)
        lr.draw_transformed_line(self.origin, self.origin+mn.Vector3(0, 0, axis_len), blue)
        # print("drawn axes ...")

    def render(self, keys=(0, "color_sensor")):
        self.sim._Simulator__sensors[keys[0]][keys[1]].draw_observation()
        arr = self.sim.get_sensor_observations()[keys[1]]
        self.draw_axes()
        arr = self.sim.get_sensor_observations()[keys[1]]
        return arr

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="/Users/rishi/Documents/Academics/stanford/contextual-character-animation/habitat-sim-mocap/floorplanner_chunks.102815835-dining.dining1_chairs_1_dining1_213.1 copy 2/Batch0/floorplanner_chunks.102815835-dining.dining1_chairs_1_dining1_213.1.reaching.0451",
        help="path to the folder with the bvh, urdf, and vr_data.json files",
    )

    args = parser.parse_args()

    experiment_dir = pathlib.Path(args.experiment_dir)
    
    bvh_path = list(experiment_dir.glob("*.bvh"))[0].as_posix()
    
    with open(experiment_dir.joinpath("vr_data.json")) as f:
        vr_data = json.load(f)

    start, end = vr_data["bvh_trim_indices"]

    sim_settings = default_sim_settings
    sim_settings["scene"] = "NONE"
    sim_settings.pop("scene_dataset_config_file")
    # sim_settings["scene_dataset_config_file"] = "default"
    env = WalkerEnvHabitat(sim_settings, bvh_path, motion_start=start, motion_end=end)
    o, info = env.reset()
    frames = [env.render()]
    # dir_ = f'gifs-habitat-sim'
    
    # if not os.path.isdir(dir_):
    #     os.mkdir(dir_)

    # path = f'{dir_}/test.gif'
    # for _ in range(1):
    #     a = np.random.rand(len(env.agent_model.joint_forces))
    #     for _ in range(1):
    #         o, r, terminated, info = env.step(a)
    #     frames.append(env.render())
    #     if terminated:
    #         break
    cv2.imwrite("gifs-habitat-sim/img.png", frames[0])
    # imageio.mimsave(path, frames)