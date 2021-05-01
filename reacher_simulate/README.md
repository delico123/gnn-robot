## reacher toy dataset

### Dependencies
- pybullet
- urdfpy
- numpy

### Code
<code>
python main.py [-h] [--num_urdf NUM_URDF] [--min_link MIN_LINK] [--max_link MAX_LINK] [--max_vel MAX_VEL] [--num_iter NUM_ITER] [--save_idx SAVE_IDX] [--render RENDER]
</code>

**optional arguments:**\
  --num_urdf NUM_URDF  dataset
  --min_link MIN_LINK  min length of each link
  --max_link MAX_LINK  max length of each link
  --max_vel MAX_VEL    max vel for ee
  --num_iter NUM_ITER  number of simulation for each reacher
  --save_idx SAVE_IDX  saved idx for dataset
  --render RENDER      if want to render bullet

### Data Structure
- list of each reacher version
- each element contains **structure** and **dynamics**
- strcture: adjacency matrix + link info
- dynamics: list of joint state + velocity command + resulting position change for each iteration
