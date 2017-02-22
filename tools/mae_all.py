import numpy as np
import os
from os.path import join, splitext, split, dirname, abspath

root_dir = abspath(join(dirname(__file__), '..'))

print "Alexnet: "
tmp_dir = join(root_dir, 'experiments/alexnet/tmp')
npys = [npy for npy in os.listdir(tmp_dir) if npy[-4:] == '.npy']
npys.sort()
for f in npys:
    mae = np.load(join(tmp_dir, f))
    print "%s: bestMAE=%.4f"%(f, mae.min())

print "---\nAgenet: "
tmp_dir = join(root_dir, 'experiments/age-net-bn/tmp')
npys = [npy for npy in os.listdir(tmp_dir) if npy[-4:] == '.npy']
npys.sort()
for f in npys:
    mae = np.load(join(tmp_dir, f))
    print "%s: bestMAE=%.4f"%(f, mae.min())