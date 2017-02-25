import sys, os, re, urllib
sys.path.append('caffe-ldl/python')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop
from caffe.proto import caffe_pb2
from os.path import join, splitext, abspath, exists, dirname, isdir, isfile
from datetime import datetime
from scipy.io import savemat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
tmp_dir = 'tmp'
tmp_dir = join(dirname(__file__), tmp_dir)
if not isdir(tmp_dir):
  os.makedirs(tmp_dir)

parser = argparse.ArgumentParser(description='AlexNet')
parser.add_argument('--ratio', type=float, required=False, default=0.6)
parser.add_argument('--gpu', type=int, required=False, default=0)
parser.add_argument('--data', type=str, required=False, default='Morph')
parser.add_argument('--tree', type=int, required=False, default=5)
parser.add_argument('--depth', type=int, required=False, default=7)
parser.add_argument('--drop', type=bool, required=False, default=False)
parser.add_argument('--nout', type=int, required=False, default=64)
args=parser.parse_args()
# read DB.info
data_source = join('data', args.data+'DB', 'Ratio'+str(args.ratio))
with open(join(data_source, 'db.info')) as db_info:
  l = db_info.readlines()[1].split(',')
  nTrain = int(re.sub("[^0-9]", "", l[0]))
  nTest = int(re.sub("[^0-9]", "", l[1]))
  minAge = int(re.sub("[^0-9]", "", l[2]))
  maxAge = int(re.sub("[^0-9]", "", l[3]))
if args.data == 'FGNet':
  test_batch_size = 50
else:
  test_batch_size = 81
train_batch_size = 64
test_iter = int(np.ceil(nTest / test_batch_size))
# some useful options ##
ntree = args.tree      #
treeDepth = args.depth #
maxIter = 30000        #
test_interval = 500    #
########################
def make_net(phase='train'):
  data_source = join('data', args.data+'DB', 'Ratio'+str(args.ratio))
  n = caffe.NetSpec()
  if phase=='train':
    batch_size = train_batch_size
  elif phase=='test':
    batch_size = test_batch_size
  if phase != 'deploy':
    n.data = L.Data(source=join(data_source, phase+'-img'), backend=P.Data.LMDB, batch_size=batch_size,
                    transform_param=dict(mean_value=112,crop_size=227), ntop=1)
    n.label = L.Data(source=join(data_source, phase+'-age'), backend=P.Data.LMDB, batch_size=batch_size, ntop=1)
  else:
    n.data = L.Input(shape=dict(dim=[1,3,256,256]))
  n.conv1 = L.Convolution(n.data, kernel_size=11, stride=4, num_output=96, weight_filler=dict(type='xavier'),
    param=[dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)])
  n.relu1 = L.ReLU(n.conv1, in_place=True)
  n.lrn1 = L.LRN(n.relu1, local_size=5, alpha=0.0001, beta=0.75)
  n.pool1 = L.Pooling(n.lrn1, pool=P.Pooling.MAX, stride=2, kernel_size=3)
  
  n.conv2 = L.Convolution(n.pool1, kernel_size=5, pad=2, num_output=256, group=2, weight_filler=dict(type='xavier'),
    param=[dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)])
  n.relu2 = L.ReLU(n.conv2, in_place=True)
  n.lrn2 = L.LRN(n.relu2, local_size=5, alpha=0.0001, beta=0.75)
  n.pool2 = L.Pooling(n.lrn2, pool=P.Pooling.MAX, stride=2, kernel_size=3)

  n.conv3 = L.Convolution(n.pool2, kernel_size=3, pad=1, num_output=384, weight_filler=dict(type='xavier'),
    param=[dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)])
  n.relu3 = L.ReLU(n.conv3, in_place=True)
  
  n.conv4 = L.Convolution(n.relu3, kernel_size=3, pad=1, num_output=384, group=2, weight_filler=dict(type='xavier'),
    param=[dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)])
  n.relu4 = L.ReLU(n.conv4, in_place=True)
  
  n.conv5 = L.Convolution(n.relu4, kernel_size=3, pad=1, num_output=256, group=2, weight_filler=dict(type='xavier'),
    param=[dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.2, decay_mult=0)])
  n.relu5 = L.ReLU(n.conv5, in_place=True)
  n.pool5 = L.Pooling(n.relu5, pool=P.Pooling.MAX, stride=2, kernel_size=3)

  n.fc6 = L.InnerProduct(n.pool5, num_output=4096, bias_term=True,
            weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.1),
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  n.relu6 = L.ReLU(n.fc6, in_place=True)
  n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

  n.fc7 = L.InnerProduct(n.drop6, num_output=4096, bias_term=True, 
            weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.1),
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  n.relu7 = L.ReLU(n.fc7, in_place=True)
  n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
  if args.nout > 0:
    assert(args.nout >= int(pow(2, treeDepth - 1) - 1))
    nout = args.nout
  else:
    if ntree == 1:
      nout = int(pow(2, treeDepth - 1) - 1)
    else:
      nout = int((pow(2, treeDepth - 1) - 1) * ntree * 2 / 3)
  n.fc8 = L.InnerProduct(n.drop7, num_output=nout, bias_term=True, weight_filler=dict(type='xavier'),
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], name='fc8a')

  if phase=='train':
    all_data_vec_length = int(nTrain / train_batch_size)
    n.loss = L.NeuralDecisionDLForestWithLoss(n.fc8, n.label, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
        neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree, num_classes=maxAge-minAge+1, iter_times_class_label_distr=20, 
        iter_times_in_epoch=100, all_data_vec_length=all_data_vec_length, drop_out=args.drop), name='probloss')
  elif phase=='test':
    n.pred = L.NeuralDecisionForest(n.fc8, n.label, neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree, num_classes=maxAge-minAge+1), name='probloss')
    n.MAE = L.MAE(n.pred, n.label)
  elif phase=='deploy':
    n.pred = L.NeuralDecisionForest(n.fc8, neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree, num_classes=maxAge-minAge+1), name='probloss')
  return n.to_proto()

def make_solver():
  s = caffe_pb2.SolverParameter()
  s.random_seed = 0xCAFFE
  s.type = 'SGD'
  s.display = 5
  s.base_lr = 1e-1
  s.lr_policy = "step"
  s.gamma = 0.5
  s.momentum = 0.9
  s.stepsize = 10000
  s.max_iter = maxIter
  s.snapshot = 5000
  snapshot_prefix = join(dirname(__file__), 'model')
  if not isdir(snapshot_prefix):
    os.makedirs(snapshot_prefix)
  s.snapshot_prefix = join(snapshot_prefix, args.data + '-Ratio' + str(args.ratio))
  s.train_net = join(tmp_dir, args.data + '-train-ratio' + str(args.ratio) + '.prototxt')
  s.test_net.append(join(tmp_dir, args.data + '-test-ratio' + str(args.ratio) + '.prototxt'))
  s.test_interval = maxIter + 1 # will test mannualy
  s.test_iter.append(test_iter)
  s.test_initialization = False
  # s.debug_info = True
  return s

if __name__ == '__main__':
  # write training/testing nets and solver
  with open(join(tmp_dir, args.data + '-train-ratio' + str(args.ratio) + '.prototxt'), 'w') as f:
    f.write(str(make_net()))
  with open(join(tmp_dir, args.data + '-test-ratio' + str(args.ratio) + '.prototxt'), 'w') as f:
    f.write(str(make_net('test')))
  with open(join(tmp_dir, args.data + '-deploy-ratio' + str(args.ratio) + '.prototxt'), 'w') as f:
    f.write(str(make_net('deploy')))
  with open(join(tmp_dir, args.data + '-solver-ratio'+str(args.ratio)+'.prototxt'), 'w') as f:
    f.write(str(make_solver()))
  if args.gpu<0:    
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
  iter = 0
  mae = []
  solver = caffe.SGDSolver(join(tmp_dir, args.data + '-solver-ratio' + str(args.ratio) + '.prototxt'))
  base_weights = join(dirname(__file__), 'bvlc_reference_caffenet.caffemodel')
  if not isfile(base_weights):
    print "Downloading base model to %s ..."%(base_weights)
    urllib.urlretrieve("http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel", base_weights)
  solver.net.copy_from(base_weights)
  print "Summarize of net parameters:"
  for p in solver.net.params:
    param = solver.net.params[p][0].data[...]
    print "  layer \"%s\":, parameter[0] mean=%f, std=%f"%(p, param.mean(), param.std())
  raw_input("Press Enter to continue...")
  while iter < maxIter:
    solver.step(test_interval)
    solver.test_nets[0].share_with(solver.net)
    mae1 = np.float32(0.0)
    for t in range(test_iter):
        mae1 += solver.test_nets[0].forward()['MAE']
    mae1 /= test_iter
    mae.append(mae1)
    iter = iter + test_interval
    print args
    print "Iter%d, currentMAE=%.4f, bestMAE=%.4f"%(iter, mae[-1], min(mae))
  mae = np.array(mae, dtype=np.float32)
  sav_fn = join(tmp_dir, "MAE-%s-Ratio%.1ftree%ddepth%dtime%s"%(
          args.data, args.ratio, ntree, treeDepth, datetime.now().strftime("M%mD%d-H%HM%MS%S")))
  np.save(sav_fn+'.npy', mae)
  mat_dict = dict({'mae':mae})
  mat_dict.update(vars(args))  # save args to .mat
  savemat(sav_fn+'.mat', mat_dict)
  plt.plot(np.array(range(0,maxIter,test_interval)), mae)
  plt.title("%s: MAE vs Iter(best=%.4f)"%(args.data, mae.min()))
  plt.savefig(sav_fn+'.eps')
  plt.savefig(sav_fn+'.png')
  plt.savefig(sav_fn+'.svg')
  print "Done! Results saved at \'"+sav_fn+"\'"
