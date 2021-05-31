# Train Structure

# python main.py --rs_conv="tree" --rs_sweep --rs_sweep_short --rs_epoch=20 --rs_bs=1 &&
# python main.py --rs_conv="tree" --rs_sweep --rs_epoch=30 --rs_bs=1

# python main.py --rs_conv="test_simple_decoder" --rs_sweep --rs_sweep_short --rs_epoch=20 --rs_bs=1 &&

python main.py --rs_conv="test_decoder" --rs_sweep --rs_sweep_short --rs_epoch=20 --rs_bs=1
