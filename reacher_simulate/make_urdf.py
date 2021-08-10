import os
import random
from numpy import fix

import urdfpy

from tools.get_urdf_info import get_base,get_ee,get_link_joint

XML_DIR = './xml'

def make_reacher_urdf(save_index, min_njoint=2, max_njoint=3, MIN=0.1, MAX=0.4, str_only=False, fix_len=False, xml_dir='./xml'):
    """
    str_only: if true, fix joint # as min_njoint (for structure reconstruction task)
    """
    XML_DIR = xml_dir

    num_joint = min_njoint if str_only else random.randint(min_njoint, max_njoint)

    # fix_len
    link_max_len = (max_njoint + 2) * (0.1 + 0.4) / 4
    lst_random_length = [0] + [random.uniform(0.0005, link_max_len) for _ in range(num_joint-1)] + [link_max_len]
    lst_random_length.sort()

    links=[]
    joints=[]
    l,j=get_base()
    links.append(l)
    joints.append(j)
    for i in range(num_joint):
        if fix_len:
            random_length = lst_random_length[i+1] - lst_random_length[i]
        else:
            random_length=MIN+(MAX-MIN)*random.random()
        l,j=get_link_joint(random_length,i)
        links.append(l)
        joints.append(j)
    temp=get_ee(num_joint)
    links.append(temp)
    #for i,joint1 in enumerate(robot.joints):
    robot=urdfpy.URDF('pybullet_reacher_robot',links,joints)

    if str_only:
        joint_dir = os.path.join(XML_DIR, 'joint_{}'.format(num_joint))
        if not os.path.exists(joint_dir):
            os.makedirs(joint_dir)
        robot.save(os.path.join(joint_dir, 'reacher_'+str(save_index)+'.urdf'))
        if num_joint < max_njoint:
            make_reacher_urdf(save_index, num_joint+1, max_njoint, MIN, MAX, str_only, fix_len, XML_DIR)
    else:
        robot.save(os.path.join(XML_DIR, 'reacher_'+str(save_index)+'.urdf'))
    
def make_test_urdf(num_joint,link):
    links=[]
    joints=[]
    l,j=get_base()
    links.append(l)
    joints.append(j)
    for i in range(num_joint):
        l,j=get_link_joint(link[i],i)
        links.append(l)
        joints.append(j)
    temp=get_ee(num_joint)
    links.append(temp)
    #for i,joint1 in enumerate(robot.joints):
    robot=urdfpy.URDF('pybullet_reacher_robot',links,joints)
    robot.save(os.path.join(XML_DIR, 'reacher_infer.urdf'))

""" Test code """

if __name__=='__main__':
    NUM=2
    for i in range(NUM):
        make_reacher_urdf(i,0.1,0.4)