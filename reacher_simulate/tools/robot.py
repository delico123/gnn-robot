import pybullet as p
import time
import random

class Reacher():
    def __init__(self,render=True):
        if render:
            self.RENDER=p.GUI
        else:
            self.RENDER=p.DIRECT
        self.pyscis=p.connect(self.RENDER)
        p.setGravity(0,0,-9.8)
        
    def load(self,path_idx):
        PATH='./xml/reacher_'+str(path_idx)+'.urdf'
        self.robot_id=p.loadURDF(PATH,useFixedBase=True) #0
        self.NumJoints = p.getNumJoints(self.robot_id) #3
        for i in range(self.NumJoints):
            p.enableJointForceTorqueSensor(self.robot_id,i)

    def move_pose(self,a):
        pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0]
        joint_state=[]
        for j in range(self.NumJoints):#self.joint_id:
            #print(p.getJointState(self.robot_id,j))
            joint_state.append(p.getJointState(self.robot_id,j)[0])       
        for i in range(self.NumJoints-1):
            p.setJointMotorControl2(self.robot_id,i,controlMode=p.VELOCITY_CONTROL,targetVelocity=a[i])
            p.stepSimulation()
        next_pos = p.getLinkState(self.robot_id,self.NumJoints-1)[0]
        dp=[0,0]
        for i in range(2):
            dp[i]=next_pos[i]-pos[i]
        #print(dp)
        return joint_state,dp
    def close(self):
        p.resetSimulation()