import os
import json

from tools.reacher_urdf_structure import get_structure

XML_DIR = './xml' #TODO

class logger():
    def __init__(self, path_idx=0, num_joint=0):
        """
        path_idx: N, num_urdf
        """
        self.memory=[]
        self.path_idx=path_idx
        self.dynamics=[]
        self.save_structure(num_joint)

    def append_dynamics(self,state,command,dp,next_pos):
        log={'state':state,'command':command,'dp':dp,'next_pos':next_pos}
        self.dynamics.append(log)

    def save_dynamics(self,idx=0):
        self.memory[idx]['dynamics']=self.dynamics
    
    def save_structure(self, num_joint):
        urdf_dir = os.path.join(XML_DIR, 'joint_{}'.format(num_joint))

        for i in range(self.path_idx):
            PATH = os.path.join(urdf_dir, 'reacher_'+str(i)+'.urdf')
            struct=get_structure(PATH)
            self.memory.append({'structure':struct})

    def save_json(self,save):
        PATH='./res/res_'+str(save)+'.json'
        with open(PATH,'w') as jf:
            json.dump(self.memory,jf,indent=4)

        