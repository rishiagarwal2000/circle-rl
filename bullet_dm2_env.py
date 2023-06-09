import pybullet
from pybullet_utils import bullet_client
import pybullet_data

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import pathlib
import json
import imageio
import magnum as mn

from fairmotion.core import motion, velocity
from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.ops.motion import cut, rotate, translate

import time

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

class BulletDeepmimicEnv():
    def __init__(self,
                bvh_path,
                motion_start=None, 
                motion_end=None,
                motion_stepper=0,
                fps=240, 
                frameskip=4,
                verbose=False,
                add_scene=False,
                urdf_path="CIRCLE_assets/subjects/amass.urdf",
                ref_urdf_path="CIRCLE_assets/subjects/amass.urdf",
                use_dumped_torques=False):
        self._p = bullet_client.BulletClient()
        self._p.setRealTimeSimulation(0)
        self._p.setTimeStep(1/fps)
        self._p.setPhysicsEngineParameter(numSolverIterations=30)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.verbose = verbose

        self.ref_offset = (-1,0,1)
        
        self.add_scene = add_scene

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
        
        self.use_dumped_torques = use_dumped_torques
        if use_dumped_torques:
            self.torques = np.loadtxt('torques2.txt')
        else:
            self.torques = []
    
    def _add_stage(self):
        self._p.setGravity(0,-9.8,0)
        
        if self.add_scene:
            N = 20
            N_loop = N // 10 + (1 if N%10 != 0 else 0)
            mapping = [504+i for i in range(N)]
            for i in range(N_loop):
                fnames = [f"floorplanner/floorplanner_{mapping[i]}.obj" for i in range(i * 10, min((i+1)*10, N))]
                sceneShapeId = self._p.createVisualShapeArray(shapeTypes=[self._p.GEOM_MESH for _ in range(len(fnames))], 
                                            fileNames=fnames, 
                                            # rgbaColor=[1, 1, 1, 1],
                                            # specularColor=[0.4, .4, 0],
                                            # visualFramePosition=shift,
                                            # meshScale=meshScale
                                        )
                sceneCollId = self._p.createCollisionShapeArray(shapeTypes=[self._p.GEOM_MESH for _ in range(len(fnames))], 
                                            fileNames=fnames, 
                                            # rgbaColor=[1, 1, 1, 1],
                                            # specularColor=[0.4, .4, 0],
                                            # visualFramePosition=shift,
                                            # meshScale=meshScale
                                        )
                sceneId = self._p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sceneCollId, baseVisualShapeIndex=sceneShapeId, baseOrientation=self._p.getQuaternionFromEuler([0,0,0]))
                print(f"sceneId={sceneId}, fnames={fnames}")

        self.stage_id = self._p.loadURDF("plane.urdf", baseOrientation=self._p.getQuaternionFromEuler([-np.pi/2,0,0]))
    
    def _load_motion(self):
        # loading text because the setup pauses here during motion load
        self.print_debug("Loading...")
        quat = self._p.getQuaternionFromEuler([0,0,0])
        # final_rotation_correction = mn.Quaternion((quat[:3], quat[3]))
        self.motion = cut(bvh.load(file=self.bvh_path), self.motion_start, self.motion_end) #, np.array(self.final_rotation_correction.to_matrix())) #cut(bvh.load(file=self.bvh_path), self.motion_start, self.motion_end)
        # self.motion = rotate(self.motion, np.array(final_rotation_correction.to_matrix()))
        # self.motion = translate(self.motion, np.array([0,1.5,0]))
        # motion_height_var = np.min(self.motion.positions(local=False), axis=1)
        # print(motion_height_var.shape, motion_height_var[0], motion_height_var[100], np.min(motion_height_var, axis=0), np.max(motion_height_var, axis=0))
        
        self.motion_with_vel = velocity.MotionWithVelocity.from_motion(self.motion)
        # self.motion_with_vel.compute_velocities()
        self.print_debug("Done Loading.")
    
    def _add_agent(self):
        startPos = [0,1.25,0]
        startOrientation = self._p.getQuaternionFromEuler([np.pi/2,0,0])

        def add_wo_motors(path):
            handle = self._p.loadURDF(self.agent_urdf_path, startPos, startOrientation, useFixedBase=False)
            jointIndices = [i for i in range(self._p.getNumJoints(handle)) if self._p.getJointInfo(handle, i)[2] == self._p.JOINT_SPHERICAL]
            self._p.setJointMotorControlMultiDofArray(
                handle,
                jointIndices,
                self._p.POSITION_CONTROL,
                targetPositions=[[0, 0, 0, 1] for _ in range(len(jointIndices))],
                targetVelocities=[[0, 0, 0] for _ in range(len(jointIndices))],
                positionGains=[0 for _ in range(len(jointIndices))],
                velocityGains=[1 for _ in range(len(jointIndices))],
                forces=[[0,0,0] for _ in range(len(jointIndices))]
            )
            return handle    
        self.agent_id = add_wo_motors(self.agent_urdf_path)
        self.ref_id = add_wo_motors(self.ref_urdf_path)


    def get_action_dim(self):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        return 3 * len(jointIndices)
    
    def pd_controller(self, tgt_pose, kp=10, kd=0.1):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        obs = self.get_observation(type_="dict")
        jpose = obs["jpose"][1:]
        jvels = obs["jvel"][2:]
        print(jpose.shape, jvels.shape, np.array(tgt_pose).shape)

        # print(f"jpose={jpose}, tgt_pose={tgt_pose}")
        print(f"diffs={[np.array(self._p.getAxisDifferenceQuaternion(rp, sp)) for (sp,rp,vel) in zip(jpose, tgt_pose, jvels)]}, vels={jvels}")
        torques = [kp * np.array(self._p.getAxisDifferenceQuaternion(rp, sp)) - kd * vel  for (sp,rp,vel) in zip(jpose, tgt_pose, jvels)]
        torques = np.array(torques).flatten()
        torques = np.clip(10 * torques, -1000, 1000)
        # print(f"torques={torques}")
        # exit(1)
        return torques

    def step(self, action=None, w={"basePos": 0.2, "jpose": 0.65, "end":0.15, "jvel": 0.0}, a={"basePos": 10, "jpose": 2, "end": 40, "vel": 0.1}):
        self.motion_stepper += 1  
        
        sim_data = self.get_observation(type_="dict")
        
        if action is None:
            basePos, baseOrn, jointOrns = self._get_ref_pose()
            action = jointOrns
        else:
            # print(f"action={action}")
            basePos, baseOrn, jointOrns = self._get_ref_pose()
            cur_pose = sim_data["jpose"][1:] # jointOrns #
            cur_pose_rotvec = [R.from_quat(c) for c in cur_pose]
            action = action.reshape(-1,3)
            # print(f"action={action}, cur_pose_rotvec={cur_pose_rotvec}")
            action = [(R.from_rotvec(action[i]) * cur_pose_rotvec[i]).as_quat() for i in range(action.shape[0])] # + cur_pose_rotvec[i]).as_quat()
        
        self.apply_position_control(action)
        
        # if not self.use_dumped_torques:
        #     self.apply_position_control()
        #     self.apply_action(action)
            # pass
        

        for f in range(self.frameskip):
            # if self.use_dumped_torques:
            #     action = self.torques[(self.motion_stepper-1)*self.frameskip + f]
            #     print(f"action={action}, ")
            #     self.apply_torque_control(action)
            # else:
            #     self.apply_action(action)
            # self.print_existing_torques()
            # computed_torques = self.pd_controller(action)
            # # print(f"computed_torques={computed_torques}")
            # self.apply_torque_control(computed_torques)
            self._p.stepSimulation()
            # self.print_existing_torques()
            # pass
        
        self.update_ref()
        
        sim_data = self.get_observation(type_="dict")
        ref_data = self.get_observation(handle=self.ref_id, type_="dict")
        
        diff = {
                "basePos": np.linalg.norm(sim_data["basePos"] - ref_data["basePos"]),
                "jpose": np.sum([np.linalg.norm(self._p.getEulerFromQuaternion(self._p.getDifferenceQuaternion(sp, rp))) 
                        for (sp, rp) in zip(sim_data["jpose"], ref_data["jpose"])]),
                "end": np.sum(np.linalg.norm(ref_data["end"] - sim_data["end"], axis=1)),
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
            if not self.use_dumped_torques:
                np.savetxt("torques2.txt", np.array(self.torques))
        # reward += not terminated
        # print(f"reward={reward}, terminated={terminated}")
        return obs, reward, terminated, False, {"diff": diff}

    def update_ref(self):
        self.set_pose(self.ref_id, ref=True)

    def apply_torque_control(self, scaled_action):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        
        self._p.setJointMotorControlMultiDofArray(
            self.agent_id, 
            jointIndices=jointIndices, 
            controlMode=self._p.TORQUE_CONTROL, 
            forces=[scaled_action[ind*3:ind*3+3] for ind in range(0, int(self.get_action_dim() / 3))]
        )
    
    def print_existing_torques(self):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        existing_torques = []
        for j in jointIndices:
            js = self._p.getJointStateMultiDof(self.agent_id, j)
            existing_torques.append(js[-1])
        print(f"existing_torques={existing_torques}")

    def apply_action(self, action):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]

        if action is not None:
            existing_torques = []
            for j in jointIndices:
                js = self._p.getJointStateMultiDof(self.agent_id, j)
                existing_torques.append(js[-1])
            
            scaled_action = np.clip(30 * action, -50, 50)
            self.apply_torque_control(scaled_action)
            new_torques = []
            for j in jointIndices:
                js = self._p.getJointStateMultiDof(self.agent_id, j)
                new_torques.append(js[-1])

            torques = np.array([e for t in new_torques for e in t]) + np.array(scaled_action)
            self.torques.append(torques)
            # print(f"action={action}, existing_torques={existing_torques}, new_torques={new_torques}, torques={torques}")

    def apply_position_control(self, tgt_pose=None):
        jointIndices = [i for i in range(self._p.getNumJoints(self.agent_id)) if self._p.getJointInfo(self.agent_id, i)[2] == self._p.JOINT_SPHERICAL]
        
        if tgt_pose is None:
            basePos, baseOrn, tgt_pose = self._get_ref_pose()

        self._p.setJointMotorControlMultiDofArray(
            self.agent_id, 
            jointIndices=jointIndices, 
            controlMode=self._p.POSITION_CONTROL, 
            targetPositions=[tgt_pose[j] for j in range(len(jointIndices))], targetVelocities=[[0.0,0.0,0.0] for _ in range(len(jointIndices))], 
            positionGains=[0.5 for _ in range(len(jointIndices))], velocityGains=[0.1 for _ in range(len(jointIndices))],
            forces=[[1000,1000,1000] for _ in range(len(jointIndices))]
        )

    def get_observation(self, handle=None, type_="array"):
        pose = []
        vel = []
        handle = self.agent_id if handle is None else handle
        
        basePos, baseOrn = self._p.getBasePositionAndOrientation(handle)
        baseLinVel, baseAngVel = self._p.getBaseVelocity(handle)
        
        pose +=  list(basePos) + list(R.from_quat(baseOrn).as_rotvec())
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
                pose += list(R.from_quat(jpos).as_rotvec())
                vel += jvel
                jposes.append(jpos)
                jvels.append(jvel)

            if "wrist" in jointName or "ankle" in jointName:
                li = self._p.getLinkState(handle, j, computeForwardKinematics=1)
                eff_loc.append(np.array(li[0]) - np.array(basePos))
        # print(f"pose: {len(pose)}, {pose}")
        basePos = tuple(np.array(basePos) - np.array(self.ref_offset)) if handle == self.ref_id else basePos
        if type_ == "array":
            return np.concatenate((pose, vel))
        elif type_ == "dict":
            return {"basePos": np.array(basePos)[[0,2]], "jpose": np.array(jposes), "jvel": np.array(jvels), "end": np.array(eff_loc)-np.array(basePos)}
        else:
            raise NotImplementedError(f"type_={type_} not found")
    
    def _get_ref_pose(self, ref=False):
        #### Constants ###
        ROOT, LAST = 0, -1
        pose = self.motion.get_pose_by_time(self.motion_stepper * 1 / self.fps * self.frameskip) #poses[self.motion_stepper]

        root_T = pose.get_transform(ROOT, local=False)
        # root_T[0:3, 3] = -root_T[0, 3], root_T[2, 3], root_T[1, 3]

        quat = self._p.getQuaternionFromEuler([0,0,0])
        final_rotation_correction = global_correction_quat(mn.Vector3.z_axis(), -mn.Vector3.y_axis()) #mn.Quaternion((quat[:3], quat[3]))
        # print(f"final_rot_correction={final_rotation_correction}")
        
        root_rotation = final_rotation_correction * mn.Quaternion.from_matrix(
            mn.Matrix3x3(root_T[0:3, 0:3])
        )
        
        root_rotation = list(root_rotation.vector) + [root_rotation.scalar]
        root_rotation = [-root_rotation[0], root_rotation[2], root_rotation[1], root_rotation[3]]
        # print(root_rotation, mn.Quaternion.from_matrix(
        #     mn.Matrix3x3(root_T[0:3, 0:3])
        # ))
        # positions = pose.to_matrix(local=False)[...,:3,3]
        # print(f"positions={positions.shape}, min={np.min(positions, axis=0)}, max={np.max(positions, axis=0)}")
        # height_correct = -np.min(positions, axis=0)[2]
        # global_translation_offset = mn.Vector3(0,0,height_correct) if not ref else mn.Vector3(0,0,height_correct) + mn.Vector3(self.ref_offset)
        # print(f'root_t={root_T[0:3, 3]}, transformed_root_t={final_rotation_correction.transform_vector(root_T[0:3, 3])}')
        global_translation_offset = mn.Vector3(0,0.02,0) if not ref else mn.Vector3(0,0.02,0) + mn.Vector3(self.ref_offset)
        root_translation = final_rotation_correction.transform_vector(root_T[0:3, 3])

        root_translation = mn.Vector3([-root_translation[0], root_translation[2], root_translation[1]]) + global_translation_offset

        # self.print_debug(f"joint names={pose.skel.index_joint.keys()}")
        targetPositions = []
        
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
        
        for j in range(self._p.getNumJoints(self.agent_id)):
            ji = self._p.getJointInfo(self.agent_id, j)
            jointType = ji[2]

            if (jointType == self._p.JOINT_SPHERICAL):
                jointName = ji[1].decode()
                if jointName not in jmap.keys():
                    continue
                pose_joint_index = pose.skel.index_joint[jmap[jointName]]

                # print(f"jointName={jointName}, linkName={ji[12]}, poseJointIndex={pose_joint_index}")
                T = pose.get_transform(pose_joint_index, local=True)
                Q, _ = conversions.T2Qp(T)
                Q = list(Q)
                Q = [-Q[0],Q[2],Q[1],Q[3]]
                # Q = [Q[1], Q[2], Q[3], Q[0]]
                # print(f"Q={Q}")
                targetPosition = Q
                # self.print_debug("spherical position: ", targetPosition)
                targetPositions.append(targetPosition)

        return list(root_translation), root_rotation, targetPositions

    def _get_ref_vel(self):
        raise NotImplementedError
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

    def reset(self, start=None):
        max_steps = int(self.fps * len(self.motion.poses) / self.motion.fps / self.frameskip)
        self.motion_stepper = start if start is not None else np.random.randint(0, int(3*max_steps/4))
        self.set_pose(self.agent_id)
        self.set_pose(self.ref_id, ref=True)
        
        sim_data = self.get_observation(type_="dict")
        ref_data = self.get_observation(handle=self.ref_id, type_="dict")
        
        diff = {
                "basePos": np.linalg.norm(sim_data["basePos"] - ref_data["basePos"]),
                "jpose": np.sum([np.linalg.norm(self._p.getEulerFromQuaternion(self._p.getDifferenceQuaternion(sp, rp))) 
                        for (sp, rp) in zip(sim_data["jpose"], ref_data["jpose"])]),
                "end": np.sum(np.linalg.norm(ref_data["end"] - sim_data["end"], axis=1)),
                # "jvel": np.sum(np.linalg.norm(ref_data["jvel"] - sim_data["jvel"], axis=1)),
            }
        for k,v in diff.items():
            assert np.isclose(v, 0, atol=1e-5), f"init failed, ref different from sim at step 0 itself, key={k}, value={v}"
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
        # baseLin, baseAng, jointAng = self._get_ref_vel()
        self._p.resetBasePositionAndOrientation(handle, basePos, baseOrn)
        # self._p.resetBaseVelocity(handle, baseLin, baseAng)
        jind = 0
        for j in range(self._p.getNumJoints(handle)):
            ji = self._p.getJointInfo(handle, j)
            jointType = ji[2]
            if (jointType == self._p.JOINT_SPHERICAL):
                self._p.resetJointStateMultiDof(handle, j, targetValue=jointOrns[jind]) #, targetVelocity=jointAng[jind])
                jind += 1

    def render(self, keys=(0, "color_sensor")):
        cam_dist = 3
        cam_yaw = 15
        cam_pitch = -15
        render_width = 500
        render_height = 500
        # basePos, baseOrn, jointOrns = self._get_ref_pose()
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.cam_pos,
                                                            distance=cam_dist,
                                                            yaw=cam_yaw,
                                                            pitch=cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=1)
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
        default="data/bvh3",
        help="path to the folder with the bvh, urdf, and vr_data.json files",
    )

    args = parser.parse_args()

    experiment_dir = pathlib.Path(args.experiment_dir)
    
    bvh_path = list(experiment_dir.glob("*.bvh"))[0].as_posix()
    
    with open(experiment_dir.joinpath("vr_data.json")) as f:
        vr_data = json.load(f)

    start, end = vr_data["bvh_trim_indices"]
    print(f"motion start={start}, end={end}")
    end = 550
    print(f"motion start={start}, end={end}")
    env = BulletDeepmimicEnv(bvh_path, motion_start=start, motion_end=end, add_scene=False)

    o, info = env.reset(0)
    frames = [env.render()]
    dir_ = f'gifs-bullet-sim-bvh1'
    
    if not os.path.isdir(dir_):
        os.mkdir(dir_)

    path = f'{dir_}/test5_scene.gif'
    start_time = time.time()
    for _ in range(300):
        # a = np.random.rand(env.get_action_dim()) / 3
        a = np.zeros(env.get_action_dim())
        # a = None
        o, r, terminated, truncated, info = env.step(a)
        print(f"r={r}, terminated={terminated}, truncated={truncated}, info={info}")
        frames.append(env.render())
        if terminated:
            break
    end_time = time.time()
    print(f"sim time={end_time - start_time}")
    imageio.imwrite(f"{dir_}/img.png", frames[0])
    imageio.mimsave(path, frames)