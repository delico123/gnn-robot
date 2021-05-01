from tools.reacher_urdf_structure import get_structure
import json

class logger():
    def __init__(self,path_idx=0):
        self.memory=[]
        self.path_idx=path_idx
        self.dynamics=[]
        self.save_structure()

    def append_dynamics(self,state,command,dp):
        log={'state':state,'command':command,'dp':dp}
        self.dynamics.append(log)

    def save_dynamics(self,idx=0):
        self.memory[idx]['dynamics']=self.dynamics
    
    def save_structure(self):
        for i in range(self.path_idx):
            PATH='./xml/reacher_'+str(i)+'.urdf'
            struct=get_structure(PATH)
            self.memory.append({'structure':struct})

    def save_json(self,save):
        PATH='./res/res_'+str(save)+'.json'
        with open(PATH,'w') as jf:
            json.dump(self.memory,jf,indent=4)

        