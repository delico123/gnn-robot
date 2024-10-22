from make_urdf import make_reacher_urdf
from simulate import simulate_p
import argparse

parser = argparse.ArgumentParser(description='reacher')
parser.add_argument('--num_urdf', type=int, default=2, help='dataset')
parser.add_argument('--min_njoint', type=int, default=2, help='min number of joints for dataset')
parser.add_argument('--max_njoint', type=int, default=3, help='max number of joints idx for dataset')
parser.add_argument('--min_link', type=float, default=0.1, help='min length of each link')
parser.add_argument('--max_link', type=float, default=0.4, help='max length of each link')

parser.add_argument('--max_pos', type=float, default=0.5, help='max pos for ee')
parser.add_argument('--num_iter', type=int, default=1e5, help='number of simulation for each reacher')
parser.add_argument('--save_idx', type=int, default=0, help='saved idx for dataset')

parser.add_argument('--render', type=int, default=0, help='if want to render bullet')

parser.add_argument('--str_only', action='store_true', help='create structure data only (no simulation)')
parser.add_argument('--dyn_only', action='store_true', help='create dynamic data only, using existing urdf')

parser.add_argument('--fix_len', action='store_true', help='fix length of reacher')

parser.add_argument('--xml_dir', type=str, default='./xml', help="dir to save urdf")


args = parser.parse_args()

if args.dyn_only:
    pass
else:
    print('making urdf file')
    for i in range(args.num_urdf):
        make_reacher_urdf(i, args.min_njoint, args.max_njoint, args.min_link, args.max_link, args.str_only, args.fix_len, args.xml_dir)
    print('made urdf file')

print('run simulation')
simulate_p(args.num_urdf,args.num_iter,args.max_pos,args.save_idx,args.render, args.str_only, args.min_njoint, args.max_njoint, args.xml_dir)

print('finished!')


"""
TODO:
"""

"""
Note:
    Current str only data: 
        # joint: [1,7]
        num_urdf=20 for each # joint
        len link: [0.1, 0.4]
"""