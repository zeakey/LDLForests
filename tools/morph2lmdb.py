import sys, argparse, scipy, lmdb, shutil, hashlib
from PIL import Image
from collections import OrderedDict
sys.path.append('caffe-ldl/python')
import caffe
import numpy as np
from random import shuffle
import scipy
import os, re
from os.path import join, splitext, split, abspath, isdir
parser = argparse.ArgumentParser(description='Convert Morph database to LMDB')
parser.add_argument('--data', type=str, help='Morph database directory', required=True)
parser.add_argument('--ratio', type=float, help='Training set ratio', required=False, default=0.5)
parser.add_argument('--imsize', type=int, help='Image size', required=False, default=256)
parser.add_argument('--std', type=float, help='gaussian std', required=False, default=10)
parser.add_argument('--debug', type=bool, help='debug', required=False, default=False)
args = parser.parse_args()
if args.debug:
  import matplotlib.pyplot as plt
NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'

def is_image(im):
  return ('.jpg' in im) or ('.JPG' in im) or ('.PNG' in im) or ('.png' in im)

max_age = max([int(re.sub("[^0-9]", "", img)[-2::]) for img in os.listdir(args.data) if is_image(img)])
min_age = min([int(re.sub("[^0-9]", "", img)[-2::]) for img in os.listdir(args.data) if is_image(img)])
mean_age = np.mean(np.array([int(re.sub("[^0-9]", "", img)[-2::]) for img in os.listdir(args.data) if is_image('.JPG')], dtype=np.float))
gaussian = scipy.signal.gaussian(max_age - min_age + 1, args.std)

def make_label(label_value):
  label_value = label_value - min_age
  label_distr = np.zeros([(max_age - min_age + 1)])
  mid = np.ceil((max_age - min_age + 1) / 2)
  shift = int(label_value - mid)
  if shift > 0:
    label_distr[shift:] = gaussian[0:-shift]
  elif shift == 0:
    label_distr = gaussian
  else:
    label_distr[:shift] = gaussian[-shift:]
  label_distr = label_distr / np.sum(label_distr)
  if args.debug:
    print "Debug Info: age=%d"%(label_value+min_age)
    plt.plot(label_distr)
    plt.show()
  return label_distr

def make_lmdb(db_path, img_list, data_type='image'):
  if os.path.exists(db_path):
    # remove the old db files, I found the old db-files would cause some error
    shutil.rmtree(db_path)
  os.makedirs(db_path)
  db = lmdb.open(db_path, map_size=int(1e12))
  with db.begin(write=True) as in_txn:
    for idx, im in enumerate(img_list):
      if data_type == 'image':
        data = np.array(Image.open(os.path.join(args.data, im)), dtype=np.float)
        data = scipy.misc.imresize(data, [args.imsize]*2)
        # data = data - 112
        data = data[:,:,::-1] # rgb to bgr
        data = data.transpose([2, 0, 1])
      elif data_type == 'age':
        age = int(re.sub("[^0-9]", "", im)[-2::])
        data = make_label(age).reshape([max_age - min_age + 1, 1, 1]).astype(np.float)
      data = caffe.io.array_to_datum(data)
      in_txn.put(IDX_FMT.format(idx), data.SerializeToString())
      if (idx+1) % 10 == 0:
        print "Serializing to %s, %d of %d, image size(%s x %s)"%(db_path, idx+1, len(img_list), args.imsize, args.imsize)
  db.close()

if __name__ == '__main__':
  img_list = [img for img in os.listdir(args.data) if is_image(img)]
  assert(len(img_list) != 0)
  # sort img_list according to md5
  img_hash = {hashlib.md5(img).hexdigest():img for img in img_list}
  img_hash = OrderedDict(sorted(img_hash.items()))
  img_list = img_hash.values()
  NTrain, NTest = int(len(img_list) * args.ratio), len(img_list) - int(len(img_list) * args.ratio)
  base_dir = abspath(join(args.data, '..', 'MorphDB'))
  # convert training data
  db_path = abspath(join(base_dir, 'Ratio'+str(args.ratio), 'train-img'))
  make_lmdb(db_path, img_list[:NTrain], 'image')
  db_path = join(base_dir, 'Ratio'+str(args.ratio), 'train-age')
  make_lmdb(db_path, img_list[:NTrain], 'age')
  ## converting testing data
  db_path = abspath(join(base_dir, 'Ratio'+str(args.ratio), 'test-img'))
  make_lmdb(db_path, img_list[NTrain:NTrain+NTest], 'image')
  db_path = abspath(join(base_dir, 'Ratio'+str(args.ratio), 'test-age'))
  make_lmdb(db_path, img_list[NTrain:NTrain+NTest], 'age')
  with open(abspath(join(base_dir, 'Ratio'+str(args.ratio),'db.info')), 'w') as db_info:
    db_info.write("Morph dataset LMDB info: TrainSet ratio=%f \n"%(args.ratio))
    db_info.write("nTrain=%d, nTest=%d, minAge=%d, maxAge=%d, meanAge=%f, imsize=%d, std=%f"\
      %(NTrain, NTest, min_age, max_age, mean_age, args.imsize, args.std))
  if not isdir("data"):
    os.makedirs("data")
  if not isdir("data/MorphDB"):
    os.symlink(base_dir, "data/MorphDB")
    print "Make data symbol link at 'data/MorphDB'."
  
