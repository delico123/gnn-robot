# gnn-robot
(2021) Graph neural networks for robotics domain


- PyTorch
- PyBullet
- W&B

Specific version: TBD


## Data Generation 
- Details and dependencies can be found in `reacher_simulate/README.md`
- To generate `structure_only` data (or run `script_data_gen.sh`)
```
# Generate data. Structure only
cd ./reacher_simulate
python main.py --str_only --num_urdf 1000 --min_njoint 2 --max_njoint 7

# Dyn after str_only created
cd ./reacher_simulate
python main.py --dyn_only --num_urdf 1000 --min_njoint 3 --max_njoint 3 --num_iter 20 --save_idx 1

# TODO: joint
```

## Run model
- Run `script_rstruc.sh`.
