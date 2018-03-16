from visdom import Visdom
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file')

args = parser.parse_args()

viz = Visdom()
iter = []
accuracy_cls = []
loss_mask = []

print(args.file)
skip_iters = 100

with open(args.file, 'r') as f:
    for line in f:
        if line.startswith("json_stats"):
            stats = json.loads(line[12:])

            if stats["iter"] < 100:
                continue

            iter.append(stats["iter"])
            accuracy_cls.append(stats["accuracy_cls"])
            loss_mask.append(stats["loss_mask"])

iter_arr = np.array(iter)
accuracy_cls_arr = np.array(accuracy_cls)
loss_mask_arr = np.array(loss_mask)

viz.line(X=iter_arr, Y=np.column_stack((accuracy_cls_arr, loss_mask_arr)))