from visdom import Visdom
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file')

args = parser.parse_args()

viz = Visdom()
iter = []
#accuracy_cls = []
loss = []

print(args.file)
skip_iters = 100

with open(args.file, 'r') as f:
    for line in f:
        if line.startswith("json_stats"):

            try:
                stats = json.loads(line[12:])
            except:
                break

            if stats["iter"] < 100:
                continue

            iter.append(stats["iter"])
            #accuracy_cls.append(stats["accuracy_cls"])
            loss.append(stats["loss"])

iter_arr = np.array(iter)
#accuracy_cls_arr = np.array(accuracy_cls)
loss_arr = np.array(loss)

viz.line(X=iter_arr, Y=loss_arr)