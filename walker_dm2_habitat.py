import habitat_sim
from habitat_sim.logging import logger
from settings import default_sim_settings, make_cfg

import habitat_sim.physics as phy
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum, angle_between_quats, quat_from_coeffs
from scipy.spatial.transform import Rotation as R

import magnum as mn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
import pathlib
import json

from fairmotion.core import motion, velocity
from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.ops.motion import cut, rotate, translate

import os
import time
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
            motion_start=None, 
            motion_end=None,
            motion_stepper=0,
            fps=1/0.0041, 
            frameskip=4,
            urdf_path="CIRCLE_assets/subjects/amass.urdf", #"human.urdf", #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/quadruped/quadruped.urdf" #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/humanoid/nao.urdf" #"CIRCLE_assets/subjects/1_agent.urdf"
            ref_urdf_path="CIRCLE_assets/subjects/amass.urdf" #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/quadruped/quadruped.urdf" #"/Users/rishi/Documents/Academics/stanford/contextual-character-animation/bullet3/data/humanoid/nao.urdf" #"CIRCLE_assets/subjects/1_agent.urdf"
    ):
        self.sim_settings = sim_settings
        self.urdf_path = urdf_path
        self.ref_urdf_path = ref_urdf_path
        
        self.bvh_path = bvh_path
        self.motion_stepper = motion_stepper
        self.motion_start: int = motion_start
        self.motion_end: int = motion_end

        self.fps = fps
        self.frameskip = frameskip
        # print("starting reconfiguring")
        self._reconfigure_sim()
        self._load_motion()
        # print("loaded motion")

        self.ref_episode_len = len(self.motion.poses)
        # print("getting art obj manager")
        self.art_obj_mgr = self.sim.get_articulated_object_manager()
        self.agent_model = None
        self.ref_model = None
        self.ref_offset = (-2, 0, 1)
        # print("adding stage")
        self._add_stage()
        # print("Added stage")
        self.reset()
        # print("Reset")
    
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
        self.motion = cut(bvh.load(file=self.bvh_path), self.motion_start, self.motion_end) #, np.array(self.final_rotation_correction.to_matrix())) #cut(bvh.load(file=self.bvh_path), self.motion_start, self.motion_end)
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
        self.stage.motion_type = phy.MotionType.DYNAMIC
        self.stage.rotation = global_correction_quat(mn.Vector3.z_axis(), mn.Vector3.x_axis()) #self.final_rotation_correction
        self.stage.translation = mn.Vector3(0,0.1,0)

    def _add_agent_model(self):
        
        if self.agent_model:
            self.art_obj_mgr.remove_object_by_handle(self.agent_model.handle)
            self.agent_model = None

        self.agent_model = self.art_obj_mgr.add_articulated_object_from_urdf(
            filepath=self.urdf_path, fixed_base=False
        )

        assert self.agent_model.is_alive

        self.agent_model.motion_type = phy.MotionType.DYNAMIC

    def _add_ref_model(self):
        
        if self.ref_model:
            self.art_obj_mgr.remove_object_by_handle(self.ref_model.handle)
            self.ref_model = None

        self.ref_model = self.art_obj_mgr.add_articulated_object_from_urdf(
            filepath=self.ref_urdf_path, fixed_base=False
        )

        assert self.ref_model.is_alive

        self.ref_model.motion_type = phy.MotionType.KINEMATIC

    def get_action_dim(self):
        return len(self.agent_model.joint_forces)

    def step(self, action=None, w={"basePos": 0.3, "jpose": 0.7, "end":0.0, "jvel": 0.0}, a={"basePos": 10, "jpose": 2, "end": 40, "vel": 0.1}):
        
        self.motion_stepper = (self.motion_stepper + 1) % len(self.motion.poses)
        # print(f"action={action}")
        self._next_ref_pose()
        posbefore = np.array(self.agent_model.translation)[2]
        
        self.update_pd_targets()

        if action is not None:
            scaled_action = np.clip(10 * action, -15, 15)
            self.agent_model.add_joint_forces(scaled_action)

        for _ in range(self.frameskip):
            self.sim.step_world()
        
        sim_data = self.get_observation(type_="dict")
        ref_data = self.get_observation(handle=self.ref_model, type_="dict")
        
        # sps = sim_data["jpose"]
        # rps = ref_data["jpose"]
        # print(f'sps={sps[0]}, rps={rps[0]}, angle={angle_between_quats(quat_from_coeffs(sps[0]), quat_from_coeffs(rps[0]))}')
        diff = {
                "basePos": np.linalg.norm(sim_data["basePos"] - ref_data["basePos"]),
                "jpose": np.sum([angle_between_quats(quat_from_coeffs(sp), quat_from_coeffs(rp)) 
                        for (sp, rp) in zip(sim_data["jpose"], ref_data["jpose"])]),
                "end": 0,
                # "jvel": np.sum(np.linalg.norm(ref_data["jvel"] - sim_data["jvel"], axis=1)),
            }

        # print(f"diff={diff}")
        reward = w["basePos"] * np.exp(-a["basePos"] * diff["basePos"]) \
                + w["jpose"] * np.exp(-a["jpose"] * diff["jpose"]) \
                + w["end"] * np.exp(-a["end"] * diff["end"])
                # + w["jvel"] * np.exp(-a["vel"] * diff["jvel"])

        obs = self.get_observation()        
        terminated = False
        
        if reward < 0.1 or (self.motion_stepper+1) * self.frameskip / self.fps > len(self.motion.poses) / self.motion.fps:
            terminated = True
        # print(f"reward={reward}, terminated={terminated}")
        return obs, reward, terminated, False, {"diff": diff}
    
    def add_pd_control(self):
        self.pd_joint_motors = []
        new_pose, new_pose_T, new_root_translate, new_root_rotation = self.get_ref_pose()
        model = self.agent_model
        count = 0
        for model_link_id in model.get_link_ids():
            joint_type = model.get_link_joint_type(model_link_id)
            if joint_type == phy.JointType.Fixed:
                continue
            elif joint_type == phy.JointType.Spherical:
                self.pd_joint_motors.append(model.create_joint_motor(model_link_id, phy.JointMotorSettings(mn.Quaternion(new_pose[count:count+3], new_pose[count+3]), 2, mn.Vector3(0,0,0), 0.1, 100)))
                count += 4
            elif joint_type == phy.JointType.Revolute:
                raise NotImplementedError()
    
    def update_pd_targets(self):
        new_pose, new_pose_T, new_root_translate, new_root_rotation = self.get_ref_pose()
        model = self.agent_model
        model.update_all_motor_targets(new_pose)
        
    def _next_ref_pose(self):
        self.set_pose(self.ref_model, ref=True)

    def get_observation(self, handle=None, type_="array"):
        pose = []
        vel = []
        handle = self.agent_model if handle is None else handle
        
        basePos, baseOrn = list(handle.translation), list(handle.rotation.vector) + [handle.rotation.scalar]
        baseLinVel, baseAngVel = list(handle.root_linear_velocity), list(handle.root_angular_velocity)
        
        pose += baseOrn + basePos
        vel += baseAngVel + baseLinVel
        
        data = handle.joint_positions
        jposes = [baseOrn] + [data[x:x+4] for x in range(0, len(data), 4)]
        data = handle.joint_velocities
        jvels = [baseLinVel, baseAngVel] + [data[x:x+3] for x in range(0, len(data), 3)]

        basePos = tuple(np.array(basePos) - np.array(self.ref_offset)) if handle == self.ref_model else basePos
        if type_ == "array":
            return np.concatenate([list(handle.rotation.vector) + [handle.rotation.scalar], handle.joint_positions, handle.joint_velocities]).ravel()
        elif type_ == "dict":
            return {"basePos": np.array(basePos)[:2], "jpose": np.array(jposes), "jvel": np.array(jvels), "end": 0}
        else:
            raise NotImplementedError(f"type_={type_} not found")
    
    def reset(self):
        max_steps = int(self.fps * len(self.motion.poses) / self.motion.fps / self.frameskip)
        self.motion_stepper = np.random.randint(0, int(3*max_steps/4))

        self._add_agent_model()
        self._add_ref_model()
        
        self.set_pose(self.agent_model)
        self.set_pose(self.ref_model, ref=True)
        self.set_camera(self.agent_model)

        self.add_pd_control()

        return self.get_observation(), {}
    
    def set_camera(self, handle):
        self.origin = handle.translation

        # print("joint positions", self.agent_model.joint_positions, self.agent_model.joint_velocities, self.agent_model.joint_forces)

        self.agent_body_node.translation = handle.translation + mn.Vector3(1, 0.8, -1)
        camera_position = self.agent_body_node.translation
        camera_look_at = handle.translation
        self.agent_body_node.rotation = mn.Quaternion.from_matrix(
            mn.Matrix4.look_at(
                camera_position, camera_look_at, np.array([0, 1.0, 0])  # up
            ).rotation()
        )

    def get_ref_pose(self, raw=True):
        """
        Returns the current pose of the model
        """
        pose = self.motion.get_pose_by_time(self.motion_stepper * 1 / self.fps * self.frameskip)
        pose_Q, pose_T, tran, rot = self.convert_CMUamass_single_pose(pose, self.ref_model, raw=raw)
        return pose_Q, pose_T, tran, rot

    def get_ref_velocity(self):
        """
        Returns the current velocity of the model
        """
        raise NotImplementedError()
        pose1_Q, pose1_T, tran1, rot1 = self.convert_CMUamass_single_pose(self.motion.poses[self.motion_stepper], self.ref_model, raw=True)
        pose2_Q, pose2_T, tran2, rot2 = self.convert_CMUamass_single_pose(self.motion.poses[(self.motion_stepper+1)%self.motion.num_frames()], self.ref_model, raw=True)
        pose1 = motion.Pose(self.motion.poses[0].skel, pose1_T)
        pose2 = motion.Pose(self.motion.poses[0].skel, pose2_T)
        vel = velocity.Velocity(pose1, pose2, dt=self.motion.fps_inv)
        return vel.data_local[1:, :3].ravel()
    
    def set_pose(self, handle, ref=False):
        new_pose, new_pose_T, new_root_translate, new_root_rotation = self.get_ref_pose()
        handle.joint_positions = new_pose
        handle.rotation = new_root_rotation
        handle.translation = new_root_translate + (mn.Vector3(self.ref_offset) if ref else mn.Vector3(0,0,0))

    def convert_CMUamass_single_pose(
        self, pose, model, raw=False
    ) -> Tuple[List[float], mn.Vector3, mn.Quaternion]:
        """
        This conversion is specific to the datasets from CMU
        """
        new_pose = []
        new_pose_T = []
        new_pose_E = []
        joint_names = []
        # Root joint
        root_T = pose.get_transform(ROOT, local=False)
        new_pose_T.append(root_T)
        
        final_rotation_correction = global_correction_quat(mn.Vector3.z_axis(), -mn.Vector3.y_axis())

        if not raw:
            final_rotation_correction = (
                global_correction_quat(mn.Vector3.y_axis(), -mn.Vector3.z_axis())
                * mn.Quaternion()
            )

        root_rotation = final_rotation_correction * mn.Quaternion.from_matrix(
            mn.Matrix3x3(root_T[0:3, 0:3])
        )

        root_rotation = list(root_rotation.vector) + [root_rotation.scalar]
        root_rotation = [-root_rotation[0], root_rotation[2], root_rotation[1], root_rotation[3]]
        root_rotation = mn.Quaternion(tuple(root_rotation[:3]), root_rotation[-1])

        root_translation = list(
            mn.Vector3(0,0,0)
            + final_rotation_correction.transform_vector(root_T[0:3, 3])
        )
        root_translation = mn.Vector3([-root_translation[0], root_translation[2], root_translation[1]])

        Q, _ = conversions.T2Qp(root_T)

        jmap = {
            "lhip": "LeftUpLeg",
            "lknee": "LeftLeg",
            "lankle": "LeftFoot",
            "rhip": "RightUpLeg",
            "rknee": "RightLeg",
            "rankle": "RightFoot", 
            "lowerback": "Spine",
            "upperback": "Spine1",
            "chest": "Spine2",
            "lowerneck": "Spine3",
            "upperneck": "Neck",
            "lclavicle": "LeftShoulder",
            "lshoulder": "LeftArm",
            "lelbow": "LeftForeArm",
            "rclavicle": "RightShoulder",
            "rshoulder": "RightArm",
            "relbow": "RightForeArm",
        }

        # Other joints
        for model_link_id in model.get_link_ids():
            joint_type = model.get_link_joint_type(model_link_id)

            if joint_type == phy.JointType.Fixed:
                continue

            joint_name = model.get_link_name(model_link_id)
            if joint_name not in jmap.keys():
                continue
            pose_joint_index = pose.skel.index_joint[jmap[joint_name]]
            
            joint_names.append(joint_name)
            # When the target joint do not have dof, we simply ignore it

            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            T = pose.get_transform(pose_joint_index, local=True)
            E = conversions.R2E(conversions.T2R(T))
            Q, _ = conversions.T2Qp(T)
            Q = list(Q)
            Q = [-Q[0],Q[2],Q[1],Q[3]]

            new_pose += Q
            new_pose_T.append(T)
            new_pose_E.append(E[np.argmax(np.abs(E))])
            
        # print(f"E_len={len(new_pose_E)}, Q_len={len(new_pose)}, new_pose_E={list(zip(new_pose_E, joint_names))}, new_pose_Q={new_pose}")
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
        default="data/bvh2",
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
    dir_ = f'gifs-habitat-sim'
    
    if not os.path.isdir(dir_):
        os.mkdir(dir_)

    path = f'{dir_}/test2.gif'
    start_sim = time.time()
    for _ in range(300): # env.ref_episode_len
        a = np.random.rand(len(env.agent_model.joint_forces))
        for _ in range(1):
            o, r, terminated, truncated, info = env.step(a)
        frames.append(env.render())
        if terminated:
            break
    print(f"sim time={time.time() - start_sim}")
    imageio.imwrite(f"{dir_}/img.png", frames[0])
    imageio.mimsave(path, frames)

    # o, info = env.reset()
    # frames = [env.render()]
    # dir_ = f'gifs-habitat-sim'
    
    # if not os.path.isdir(dir_):
    #     os.mkdir(dir_)

    # path = f'{dir_}/test3.gif'
    # for _ in range(100):
    #     a = np.random.rand(len(env.agent_model.joint_forces))
    #     for _ in range(1):
    #         o, r, terminated, truncated, info = env.step(a)
    #     frames.append(env.render())
    #     if terminated:
    #         break
    # # imageio.imwrite("gifs-habitat-sim/img.png", frames[0])
    # imageio.mimsave(path, frames)