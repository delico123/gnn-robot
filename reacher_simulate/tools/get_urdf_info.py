import urdfpy
import random

JOINT_TYPE='revolute'
AXIS=(0,0,1)

def get_base():
    geometry=urdfpy.Geometry(cylinder=urdfpy.Cylinder(0.03,0.05))
    material=urdfpy.Material('yellow',color=(0.8,0.8,0,1))
    origin=((1,0,0,0),(0,1,0,0),(0,0,1,0.025),(0,0,0,1))
    base_visual=urdfpy.Visual(geometry,material=material,origin=origin)

    inertia=((1,0,0),(0,1,0),(0,0,1))
    mass=1
    base_inertia=urdfpy.Inertial(mass,inertia,origin)

    base_collision = urdfpy.Collision(name=None,geometry=geometry,origin=origin)
    
    base_link=urdfpy.Link('link0',base_inertia,[base_visual],[base_collision])
    
    limit=urdfpy.JointLimit(600,400-3.14,3.14)
    base_joint=urdfpy.Joint('joint1', JOINT_TYPE, 'link0', 'link1', axis=AXIS, origin=origin, limit=limit)
    
    return base_link,base_joint

def get_link_joint(random_length,index):
    cyc=urdfpy.Cylinder(0.015,random_length)
    geometry=urdfpy.Geometry(cylinder=cyc)

    if index==0:
        material=urdfpy.Material('blue',color=(0,0,0.8,1))
    elif index==1:
        material=urdfpy.Material('sky',color=(0,0.8,0.8,1))
    else:
        material=urdfpy.Material('green',color=(0,0.8,0,1))
    
    origin=((0,0,1,random_length/2),(0,1,0,0),(-1,0,0,0),(0,0,0,1))
    base_visual=urdfpy.Visual(geometry,material=material,origin=origin)

    inertia=((0,0,0),(0,0,0),(0,0,0))
    mass=1
    base_inertia=urdfpy.Inertial(mass,inertia,origin)
    
    base_collision = urdfpy.Collision(name=None,geometry=geometry,origin=origin)
    
    base_link=urdfpy.Link('link'+str(index+1),base_inertia,[base_visual],[base_collision])
    
    joint_origin = ((1,0,0,random_length),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    limit=urdfpy.JointLimit(300,200,-2.96705,2.96705)
    base_joint=urdfpy.Joint('joint'+str(index+2), JOINT_TYPE, 'link'+str(index+1), 'link'+str(index+2), \
                                axis=AXIS, origin=joint_origin, limit=limit)
    return base_link,base_joint

def get_ee(num_joint):
    sph=urdfpy.Sphere(0.02)
    geometry=urdfpy.Geometry(sphere=sph)

    material=urdfpy.Material('red',color=(0.8,0,0,1))
    origin = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    base_visual=urdfpy.Visual(geometry,material=material,origin=origin)

    inertia=((1,0,0),(0,1,0),(0,0,1))
    mass=1
    base_inertia=urdfpy.Inertial(mass,inertia,origin)
    
    base_collision = urdfpy.Collision(name=None,geometry=geometry,origin=origin)
    
    base_link=urdfpy.Link('link'+str(num_joint+1),base_inertia,[base_visual],[base_collision])
    return base_link