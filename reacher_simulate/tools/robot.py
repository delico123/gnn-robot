import pybullet as p
import time
import random
import math
import numpy as np

PI = math.pi
class Reacher():
    def __init__(self,render=True,infer=False):
        if render:
            self.RENDER=p.GUI
        else:
            self.RENDER=p.DIRECT
        self.pyscis=p.connect(self.RENDER)
        self.inference_phase = infer
        p.setGravity(0,0,-9.8)
        self.width = 720
        self.height = 720
        self.fov = 40
        self.aspect = self.width / self.height
        self.near = 0.2
        self.far = 10
        self.view_matrix = p.computeViewMatrix([0.0, 0, 4.0], [0, .0, -1], [1, 0, 0])
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        
    def load(self, urdf_path):
        # PATH='./xml/reacher_'+str(path_idx)+'.urdf' # Modified: receive urdf path explicitly (path_idx -> urdf_path)
        PATH = urdf_path

        self.robot_id=p.loadURDF(PATH,useFixedBase=True,flags=p.URDF_USE_SELF_COLLISION) #0
        self.NumJoints = p.getNumJoints(self.robot_id) #3
        for i in range(self.NumJoints):
            p.enableJointForceTorqueSensor(self.robot_id,i)
        self.length = p.getLinkState(self.robot_id,self.NumJoints-1)[0][0]
        self.ori = p.getLinkState(self.robot_id,self.NumJoints-1)[1]

    def infer_load(self, urdf_path):
        # PATH='./xml/reacher_'+str(path_idx)+'.urdf' # Modified: receive urdf path explicitly (path_idx -> urdf_path)
        PATH = urdf_path

        self.robot_id_infer=p.loadURDF(PATH,useFixedBase=True,basePosition=[0,0,0],
                                        flags=p.URDF_USE_SELF_COLLISION) #0

    def move_pose(self):
        l = random.random()*self.length
        angle = random.random()*PI*2
        target = [l*math.cos(angle),l*math.sin(angle),0.025]
        joint_state=p.calculateInverseKinematics(self.robot_id,self.NumJoints-1,target,self.ori)
        for i in range(self.NumJoints-1):
            p.setJointMotorControl2(self.robot_id,i,controlMode=p.POSITION_CONTROL,targetPosition=joint_state[i])
        for _ in range(50):
            p.stepSimulation()
        # print(p.getContactPoints(self.robot_id))
        # if p.getContactPoints(self.robot_id) == ():
        collsion= False
        # else:
            # collsion = True
        pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0] 
        return joint_state, (pos[0],pos[1]), collsion
    
    def get_current_pos(self):
        pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0]
        return [pos[0],pos[1]]
    
    def move_joint(self,motion):
        for i in range(self.NumJoints-1):
            p.setJointMotorControl2(self.robot_id,i,controlMode=p.POSITION_CONTROL,targetPosition=motion[0][i,0])
            p.setJointMotorControl2(self.robot_id,i,controlMode=p.POSITION_CONTROL,targetPosition=motion[1][i,0])
        for _ in range(50):
            p.stepSimulation()
    
    def make_maker(self,dp):
        pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0]
        base = [pos[0]+dp[0],pos[1]+dp[1],pos[2]]
        if self.inference_phase:
            self.maker_id = p.loadURDF('./xml/marker.urdf',base)
    def get_camera(self):
        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        images = p.getCameraImage(self.width,self.height,self.view_matrix,projection_matrix,shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_opengl = np.reshape(images[2], (self.height, self.width, 4))
        return rgb_opengl[:,:,:3]

    def close(self):
        p.resetSimulation()