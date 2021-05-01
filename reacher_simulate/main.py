from make_urdf import make_reacher_urdf
from simulate import simulate_p
import argparse

parser = argparse.ArgumentParser(description='reacher')
parser.add_argument('--num_urdf', type=int,default=2,help='dataset')
parser.add_argument('--min_link', type=float,default=0.1,help='min length of each link')
parser.add_argument('--max_link', type=float,default=0.4,help='max length of each link')
parser.add_argument('--max_vel', type=float,default=1,help='max vel for ee')
parser.add_argument('--num_iter', type=int,default=1e5,help='number of simulation for each reacher')
parser.add_argument('--save_idx', type=int,default=0,help='saved idx for dataset')
parser.add_argument('--render', type=int,default=0,help='if want to render bullet')
args = parser.parse_args()

print('making urdf file')
for i in range(args.num_urdf):
    make_reacher_urdf(i,args.min_link,args.max_link)
print('made urdf file')
simulate_p(args.num_urdf,args.num_iter,args.max_vel,args.save_idx,args.render)
print('finished!')