# Generate data. Structure
cd ./reacher_simulate

python main.py --str_only --num_urdf 1000 --min_njoint 2 --max_njoint 7 --xml_dir="./xml"
python main.py --dyn_only --num_urdf 100 --num_iter 100 --min_njoint 2 --max_njoint 7 --xml_dir="./xml"


# python main.py --str_only --num_urdf 1000 --min_njoint 2 --max_njoint 4 --xml_dir="./xml_fix_half" --fix_len
# python main.py --dyn_only --num_urdf 100 --num_iter 100 --min_njoint 2 --max_njoint 4 --xml_dir="./xml_fix_half" --fix_len