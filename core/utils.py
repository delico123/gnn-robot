#utils.py **not used**

#TODO: data util 옮기기

node_mltr = []
childs = [
    [2,3,4], [5,6], [], [7], [],
    [], [8,9], [], []
]
for child in childs:
    node = {
        'childs': child,
        'f': random.randint(1,9)
    }
    node_mltr += node
    

def _multi_traversal(node_mltr, j):
    """
    """
    if len(j) == 0:
        return f, node_mltr

    f0 = node_mltr[j].f # current value

    # to childs
    childs = node_mltr[j].childs
    f_vals = defaultlist0
    f_vals[1] = f0
    for child in childs:
        print(f"[mltr] forward {j} -> {child}")
        f1, node_mltr = _multi_traversal(node_mltr, child)
        f_vals[child+1] = f1


    return f, node_mltr