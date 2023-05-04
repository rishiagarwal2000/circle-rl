import pybullet
from pybullet_utils import bullet_client
import pybullet_data

import os
import numpy as np
import pathlib
import json
import imageio
import magnum as mn

from fairmotion.core import motion, velocity
from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.ops.motion import cut, rotate, translate


class BulletDeepmimicEnv():
    def __init__(self,
                bvh_path,
                motion_start=None, 
                motion_end=None,
                motion_stepper=0,
                fps=240, 
                frameskip=4,
                verbose=False,
                urdf_path="CIRCLE_assets/subjects/2.urdf",
                ref_urdf_path="CIRCLE_assets/subjects/2.urdf"):
        self._p = bullet_client.BulletClient()
        self._p.setRealTimeSimulation(0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.verbose = verbose

        self.ref_offset = (1,1,0)

        self.bvh_path = bvh_path
        self.motion_stepper = motion_stepper
        self.motion_start: int = motion_start
        self.motion_end: int = motion_end

        self.fps = fps
        self.frameskip = frameskip
        
        self._load_motion()

        self.ref_episode_len = len(self.motion.poses)
        
        self.agent_id = None
        self.stage_id = None
        # print("adding stage")
        self._add_stage()
        self.agent_urdf_path = urdf_path
        self.ref_urdf_path = ref_urdf_path
        self._add_agent()
        # print("Added stage")
        self.reset()
    
    def _add_stage(self):
        self._p.setGravity(0,0,-9.8)
        self.stage_id = self._p.loadURDF("plane.urdf")
    
    def _load_motion(self):
        # loading text because the setup pauses here during motion load
        self.print_debug("Loading...")
        quat = self._p.getQuaternionFromEuler([np.pi/2,0,0])
        final_rotation_correction = mn.Quaternion((quat[:3], quat[3]))
        self.motion = bvh.load(file=self.bvh_path) #, np.array(self.final_rotation_correction.to_matrix())) #cut(bvh.load(file=self.bvh_path), self.motion_start, self.motion_end)
        self.motion = rotate(self.motion, np.array(final_rotation_correction.to_matrix()))
        # self.motion = translate(self.motion, np.array([0,1.5,0]))
        # motion_height_var = np.min(self.motion.positions(local=False), axis=1)
        # print(motion_height_var.shape, motion_height_var[0], motion_height_var[100], np.min(motion_height_var, axis=0), np.max(motion_height_var, axis=0))
        
        self.motion_with_vel = velocity.MotionWithVelocity.from_motion(self.motion)
        # self.motion_with_vel.compute_velocities()
        self.print_debug("Done Loading.")
    
    def _add_agent(self):
        startPos = [0,0,1.25]
        startOrientation = self._p.getQuaternionFromEuler([0,0,0])    
        self.agent_id = self._p.loadURDF(self.agent_urdf_path, startPos, startOrientation)
        self.ref_id = self._p.loadURDF(self.ref_urdf_path, startPos, startOrientation)

        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        self._p.setJointMotorControlMultiDofArray(
          self.ref_id,
          jointIndices,
          self._p.POSITION_CONTROL,
          targetPositions=[[0, 0, 0, 1] for _ in range(len(jointIndices))],
          targetVelocities=[[0, 0, 0] for _ in range(len(jointIndices))],
          positionGains=[0 for _ in range(len(jointIndices))],
          velocityGains=[1 for _ in range(len(jointIndices))]
          )
        #   forces=[[jointFrictionForce, jointFrictionForce, 0] for _ in range(len(jointIndices))])


    def get_action_dim(self):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        return 3 * len(jointIndices)

    def step(self, action=None, w={"basePos": 0.1, "jpose": 0.65, "end":0.15, "jvel": 0.1}, a={"basePos": 10, "jpose": 2, "end": 40, "vel": 0.1}):
        self.motion_stepper = self.motion_stepper+1
        
        self.apply_action(action)
        self.apply_position_control()
        
        for _ in range(self.frameskip):
            self._p.stepSimulation()
        
        self.update_ref()
        
        sim_data = self.get_observation(type_="dict")
        ref_data = self.get_observation(handle=self.ref_id, type_="dict")
        
        diff = {
                "basePos": np.linalg.norm(sim_data["basePos"] - ref_data["basePos"]),
                "jpose": np.sum([np.linalg.norm(self._p.getEulerFromQuaternion(self._p.getDifferenceQuaternion(sp, rp))) 
                        for (sp, rp) in zip(sim_data["jpose"], ref_data["jpose"])]),
                "end": np.sum(np.linalg.norm(ref_data["end"] - sim_data["end"], axis=1)),
                "jvel": np.sum(np.linalg.norm(ref_data["jvel"] - sim_data["jvel"], axis=1)),
            }
        
        # print(f"diff={diff}")
        
        reward = w["basePos"] * np.exp(-a["basePos"] * diff["basePos"]) \
                + w["jpose"] * np.exp(-a["jpose"] * diff["jpose"]) \
                + w["end"] * np.exp(-a["end"] * diff["end"]) \
                + w["jvel"] * np.exp(-a["vel"] * diff["jvel"])
                
        obs = self.get_observation()
        
        terminated = False
        if reward < 0.1 or (self.motion_stepper+1) * self.frameskip / self.fps > len(self.motion.poses) / self.motion.fps:
            terminated = True
        # reward += not terminated
        # print(f"reward={reward}, terminated={terminated}")
        return obs, reward, terminated, False, {}

    def update_ref(self):
        self.set_pose(self.ref_id, ref=True)

    def apply_action(self, action):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]

        if action is not None:
            self._p.setJointMotorControlMultiDofArray(
                self.agent_id, 
                jointIndices=jointIndices, 
                controlMode=self._p.TORQUE_CONTROL, 
                forces=[action[ind*3:ind*3+3] for ind in range(0, int(self.get_action_dim() / 3))]
            )

    def apply_position_control(self):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        basePos, baseOrn, jointOrns = self._get_ref_pose()

        self._p.setJointMotorControlMultiDofArray(
            self.agent_id, 
            jointIndices=jointIndices, 
            controlMode=self._p.POSITION_CONTROL, 
            targetPositions=[jointOrns[j] for j in range(len(jointIndices))], targetVelocities=[[0.0,0.0,0.0] for _ in range(len(jointIndices))], 
            positionGains=[0.1 for _ in range(len(jointIndices))], velocityGains=[0.0 for _ in range(len(jointIndices))]
        )

    def get_observation(self, handle=None, type_="array"):
        pose = []
        vel = []
        handle = self.agent_id if handle is None else handle
        
        basePos, baseOrn = self._p.getBasePositionAndOrientation(handle)
        baseLinVel, baseAngVel = self._p.getBaseVelocity(handle)
        
        pose += baseOrn + basePos
        vel += baseAngVel + baseLinVel
        
        jposes = [baseOrn]
        jvels = [baseLinVel, baseAngVel]
        eff_loc = []
        for j in range(self._p.getNumJoints(handle)):
            jointState = self._p.getJointStateMultiDof(handle, j)
            jointInfo = self._p.getJointInfo(handle, j)
            jointName = jointInfo[1].decode()

            if jointInfo[2] == self._p.JOINT_SPHERICAL:
                jpos, jvel = jointState[:2]
                pose += jpos
                vel += jvel
                jposes.append(jpos)
                jvels.append(jvel)

            elif jointInfo[2] == self._p.JOINT_FIXED:
                li = self._p.getLinkState(handle, j, computeForwardKinematics=1)
                eff_loc.append(np.array(li[0]) - np.array(basePos))

        basePos = tuple(np.array(basePos) - np.array(self.ref_offset)) if handle == self.ref_id else basePos
        if type_ == "array":
            return np.concatenate((pose, vel))
        elif type_ == "dict":
            return {"basePos": np.array(basePos)[:2], "jpose": np.array(jposes), "jvel": np.array(jvels), "end": np.array(eff_loc)-np.array(basePos)}
        else:
            raise NotImplementedError(f"type_={type_} not found")
    
    def _get_ref_pose(self, ref=False):
        #### Constants ###
        ROOT, LAST = 0, -1
        pose = self.motion.get_pose_by_time(self.motion_stepper * 1 / self.fps * self.frameskip) #poses[self.motion_stepper]

        root_T = pose.get_transform(ROOT, local=False)
        
        quat = self._p.getQuaternionFromEuler([0,0,0])
        final_rotation_correction = mn.Quaternion((quat[:3], quat[3]))
        # print(f"final_rot_correction={final_rotation_correction}")
        
        root_rotation = final_rotation_correction * mn.Quaternion.from_matrix(
            mn.Matrix3x3(root_T[0:3, 0:3])
        )
        root_rotation = list(root_rotation.vector) + [root_rotation.scalar]
        # print(root_rotation, mn.Quaternion.from_matrix(
        #     mn.Matrix3x3(root_T[0:3, 0:3])
        # ))
        positions = pose.to_matrix(local=False)[...,:3,3]
        # print(f"positions={positions.shape}, min={np.min(positions, axis=0)}, max={np.max(positions, axis=0)}")
        height_correct = -np.min(positions, axis=0)[2]
        global_translation_offset = mn.Vector3(0,0,height_correct) if not ref else mn.Vector3(0,0,height_correct) + mn.Vector3(self.ref_offset)
        # print(f'root_t={root_T[0:3, 3]}, transformed_root_t={final_rotation_correction.transform_vector(root_T[0:3, 3])}')
        root_translation = (
            global_translation_offset
            + final_rotation_correction.transform_vector(root_T[0:3, 3])
        )

        # self.print_debug(f"joint names={pose.skel.index_joint.keys()}")
        targetPositions = []
        
        for j in range(self._p.getNumJoints(self.agent_id)):
            ji = self._p.getJointInfo(self.agent_id, j)
            jointType = ji[2]

            if (jointType == self._p.JOINT_SPHERICAL):
                jointName = ji[1].decode()
                # print(f"jointName={jointName}, linkName={ji[12]}")
                pose_joint_index = pose.skel.index_joint[jointName]
                # print(f"jointName={jointName}, linkName={ji[12]}, poseJointIndex={pose_joint_index}")
                T = pose.get_transform(pose_joint_index, local=True)
                Q, _ = conversions.T2Qp(T)
                Q = list(Q)
                # Q = [Q[1], Q[2], Q[3], Q[0]]
                # print(f"Q={Q}")
                targetPosition = Q
                # self.print_debug("spherical position: ", targetPosition)
                targetPositions.append(targetPosition)
            
        return list(root_translation), root_rotation, targetPositions

    def _get_ref_vel(self):
        ROOT, LAST = 0, -1
        pose = self.motion.get_pose_by_time(self.motion_stepper * 1 / self.fps * self.frameskip)
        vel = self.motion_with_vel.get_velocity_by_time(self.motion_stepper * 1 / self.fps * self.frameskip)
        baseLin = vel.get_linear(ROOT, local=False)
        baseAng = vel.get_angular(ROOT, local=False)
        jointAng = []
        for j in range(self._p.getNumJoints(self.agent_id)):
            ji = self._p.getJointInfo(self.agent_id, j)
            jointType = ji[2]

            if (jointType == self._p.JOINT_SPHERICAL):
                jointName = ji[1].decode()
                # print(f"jointName={jointName}, linkName={ji[12]}")
                pose_joint_index = pose.skel.index_joint[jointName]
                jointAng.append(vel.get_angular(pose_joint_index, local=True))
        
        return baseLin, baseAng, jointAng

    def print_min_pos(self):
        link = None
        min_pos = np.array([100,100,100])
        for j in range(self._p.getNumJoints(self.ref_id)):
            ji = self._p.getJointInfo(self.ref_id, j)
            jointType = ji[2]

            if (jointType == self._p.JOINT_SPHERICAL):
                li = self._p.getLinkState(self.ref_id, j, computeForwardKinematics=1)
                if li[0][1] < min_pos[1]:
                    link = ji[1].decode()
                min_pos = np.minimum(min_pos, li[0])
        basePos, baseOrn, jointOrns = self._get_ref_pose(ref=True)
        print(f"min_pos={min_pos}, {basePos}, {self._p.getBasePositionAndOrientation(self.ref_id)},  linkName={link}")

    def reset(self):
        self.motion_stepper = np.random.randint(0, int(len(self.motion.poses)/2))
        self.set_pose(self.agent_id)
        self.set_pose(self.ref_id, ref=True)
        
        sim_data = self.get_observation(type_="dict")
        ref_data = self.get_observation(handle=self.ref_id, type_="dict")
        
        diff = {
                "basePos": np.linalg.norm(sim_data["basePos"] - ref_data["basePos"]),
                "jpose": np.sum([np.linalg.norm(self._p.getEulerFromQuaternion(self._p.getDifferenceQuaternion(sp, rp))) 
                        for (sp, rp) in zip(sim_data["jpose"], ref_data["jpose"])]),
                "end": np.sum(np.linalg.norm(ref_data["end"] - sim_data["end"], axis=1)),
                "jvel": np.sum(np.linalg.norm(ref_data["jvel"] - sim_data["jvel"], axis=1)),
            }
        for v in diff.values():
            assert np.isclose(v, 0), f"init failed, ref different from sim at step 0 itself"
        # print(f"diff={diff}")
        # self.print_min_pos()

        basePos, baseOrn, jointOrns = self._get_ref_pose(ref=False)
        self.cam_pos = basePos

        sim_data = self.get_observation(type_="dict")
        ref_data = self.get_observation(handle=self.ref_id, type_="dict")
        
        # print(f"sim_data={sim_data}, ref_data={ref_data}")
        # exit(1)
        return self.get_observation(), {}
    
    def set_pose(self, handle, ref=False):
        basePos, baseOrn, jointOrns = self._get_ref_pose(ref=ref)
        baseLin, baseAng, jointAng = self._get_ref_vel()
        self._p.resetBasePositionAndOrientation(handle, basePos, baseOrn)
        self._p.resetBaseVelocity(handle, baseLin, baseAng)
        jind = 0
        for j in range(self._p.getNumJoints(handle)):
            ji = self._p.getJointInfo(handle, j)
            jointType = ji[2]
            if (jointType == self._p.JOINT_SPHERICAL):
                self._p.resetJointStateMultiDof(handle, j, targetValue=jointOrns[jind], targetVelocity=jointAng[jind])
                jind += 1

    def render(self, keys=(0, "color_sensor")):
        cam_dist = 3
        cam_yaw = -15
        cam_pitch = -10
        render_width = 320
        render_height = 240
        # basePos, baseOrn, jointOrns = self._get_ref_pose()
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.cam_pos,
                                                            distance=cam_dist,
                                                            yaw=cam_yaw,
                                                            pitch=cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(render_width) /
                                                    render_height,
                                                    nearVal=0.1,
                                                    farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=render_width,
                                            height=render_height,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        return px

    def print_debug(self, *args):
        if self.verbose:
            print(*args)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="data/floorplanner_chunks.102815835-dining.dining1_chairs_1_dining1_213.1.reaching.0451",
        help="path to the folder with the bvh, urdf, and vr_data.json files",
    )

    args = parser.parse_args()

    experiment_dir = pathlib.Path(args.experiment_dir)
    
    bvh_path = list(experiment_dir.glob("*.bvh"))[0].as_posix()
    
    with open(experiment_dir.joinpath("vr_data.json")) as f:
        vr_data = json.load(f)

    start, end = vr_data["bvh_trim_indices"]

    
    env = BulletDeepmimicEnv(bvh_path, motion_start=start, motion_end=end)

    o, info = env.reset()
    frames = [env.render()]
    dir_ = f'gifs-bullet-sim'
    
    if not os.path.isdir(dir_):
        os.mkdir(dir_)

    path = f'{dir_}/test5.gif'
    for _ in range(500):
        # a = np.random.rand(env.get_action_dim()) / 3
        a = None
        o, r, terminated, truncated, info = env.step(a)
        print(f"r={r}, terminated={terminated}, truncated={truncated}, info={info}")
        frames.append(env.render())
        if terminated:
            break
    imageio.imwrite("gifs-bullet-sim/img.png", frames[0])
    imageio.mimsave(path, frames)