from urdfpy import URDF
import numpy as np
import random

def get_matrix(PATH=None):
    #PATH='./xml/reacher.urdf'
    robot = URDF.load(PATH)
    num_joint=len(robot.joints)
    adj=np.eye(num_joint)
    link_length=np.zeros((num_joint,num_joint))
    node_feat = []
    for i,joint1 in enumerate(robot.joints):
        parent1 = joint1.parent
        child1 = joint1.child
        for j,joint2 in enumerate(robot.joints):
            if i>j:
                pass
            else:
                parent2 = joint2.parent
                child2 = joint2.child
                if child1==parent2:
                    adj[i,j]=1
                    adj[j,i]=1
                    link_length[i,j]=joint2.origin[0][3].copy()
                    link_length[j,i]=joint2.origin[0][3].copy()
        
        # node feat: [end effector]
        endeff = 1 if "red" in robot.links[i+1].visuals[0].material.name else 0
        node_feat.append([endeff])

    return adj.tolist(),link_length.tolist(), node_feat

def get_structure(PATH):
    adj, link_info, node_feat = get_matrix(PATH)
    structure = {'adj':adj, 'link_info':link_info, 'node_feat': node_feat, 'urdf': PATH}
    return structure

