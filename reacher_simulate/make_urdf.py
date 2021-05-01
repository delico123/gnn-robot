import urdfpy
import random
from tools.get_urdf_info import get_base,get_ee,get_link_joint

def make_reacher_urdf(save_index,MIN,MAX):
    num_joint=random.randint(2,3)
    links=[]
    joints=[]
    l,j=get_base()
    links.append(l)
    joints.append(j)
    for i in range(num_joint):
        random_length=MIN+(MAX-MIN)*random.random()
        l,j=get_link_joint(random_length,i)
        links.append(l)
        joints.append(j)
    temp=get_ee(num_joint)
    links.append(temp)
    #for i,joint1 in enumerate(robot.joints):
    robot=urdfpy.URDF('pybullet_reacher_robot',links,joints)
    robot.save('./xml/reacher_'+str(save_index)+'.urdf')
    
if __name__=='__main__':
    NUM=2
    for i in range(NUM):
        make_reacher_urdf(i,0.1,0.4)