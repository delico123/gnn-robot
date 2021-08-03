import pybullet as p
import time
import random
import math
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
        
    def load(self, urdf_path):
        # PATH='./xml/reacher_'+str(path_idx)+'.urdf' # Modified: receive urdf path explicitly (path_idx -> urdf_path)
        PATH = urdf_path

        self.robot_id=p.loadURDF(PATH,useFixedBase=True,flags=p.URDF_USE_SELF_COLLISION) #0
        self.NumJoints = p.getNumJoints(self.robot_id) #3
        for i in range(self.NumJoints):
            p.enableJointForceTorqueSensor(self.robot_id,i)

    def move_pose(self):
        joint_state=[]   
        for i in range(self.NumJoints-1):
            joint = 2*PI*(random.random()-PI)
            p.setJointMotorControl2(self.robot_id,i,controlMode=p.POSITION_CONTROL,targetPosition=joint)
            joint_state.append(joint)
        for _ in range(10):
            p.stepSimulation()
        # print(p.getContactPoints(self.robot_id))
        if p.getContactPoints(self.robot_id) == ():
            collsion= False
        else:
            collsion = True
        pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0] 
        return joint_state, (pos[0],pos[1]),collsion
    
    def get_current_pos():
        pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0]
        return [pos[0],pos[1]]
    
    def move_joint(self,joints):
        for i in range(self.NumJoints-1):
            p.setJointMotorControl2(self.robot_id,i,controlMode=p.POSITION_CONTROL,targetPosition=joints[i])
        for _ in range(10):
            p.stepSimulation()
    
    def make_maker(self,dp):
        pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0]
        base = [pos[0]+dp[0],pos[1]+dp[1],pos[2]]
        if self.inference_phase:
            self.maker_id = p.loadURDF('./xml/marker.urdf',base)

    def close(self):
        p.resetSimulation()