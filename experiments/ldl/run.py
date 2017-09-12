import sys, os
sys.path.insert(0, 'caffe-ldl/python')
sys.path.insert(0, 'python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import numpy as np
from os.path import join, realpath, dirname 
import os
import scipy.io as io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

temp_dir = join(dirname(realpath(__file__)), 'tmp')
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

def make_net(dataset, idx):
    # get dataset size to determine batch-size&input-dimmension
    mat = io.loadmat(join('data/ldl/DataSets/',dataset+'.mat'))
    data = mat['features']
    label = mat['labels']
    N, _ = data.shape 
    assert(label.shape[0])
    N = int(np.floor(N / 10) * 10)
    NTrain = int(N*0.9)
    bsize = 1000
    n = caffe.NetSpec()
    data_layer_params = dict(batch_size=bsize,split_idx=idx,phase='train',db_name=dataset)
    n.data, n.label = L.Python(module='ldl_data_layer', layer = 'LDLDataLayer', ntop=2, param_str=str(data_layer_params))
    n.ip1 = L.InnerProduct(n.data, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
        inner_product_param=dict(num_output=512, weight_filler=dict(type='xavier')))
    
    n.loss = L.NeuralDecisionDLForestWithLoss(n.ip1, n.label, name = 'probloss',
          param=[dict(lr_mult=0), dict(lr_mult=0)],
          neural_decision_forest_param=dict(depth=5,num_trees=10,iter_times_in_epoch=50 ,iter_times_class_label_distr=20,
          all_data_vec_length=int(np.ceil(NTrain/bsize)),
          record_filename=join(temp_dir, dataset+'_lb_distr.txt')))
    
    train_net =  n.to_proto()
    
    n = caffe.NetSpec()
    data_layer_params = dict(batch_size=N/10,split_idx=idx,phase='test',db_name=dataset)
    n.data, n.label = L.Python(module='ldl_data_layer', layer = 'LDLDataLayer', ntop=2, param_str=str(data_layer_params))
    n.ip1 = L.InnerProduct(n.data, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
        inner_product_param=dict(num_output=512, weight_filler=dict(type='xavier')))
    n.pred = L.NeuralDecisionForest(n.ip1, n.label, name = 'probloss',
        param=[dict(lr_mult=0), dict(lr_mult=0)],
        neural_decision_forest_param=dict(depth=5,num_trees=10))
    n.KLD = L.KLD(n.pred, n.label)
    test_net =  n.to_proto()

    with open(join(dirname(realpath(__file__)), 'train.prototxt'), 'w') as f:
        f.write(str(train_net))
    with open(join(dirname(realpath(__file__)), 'test_kld.prototxt'), 'w') as f:
        f.write(str(test_net))
    
def make_solver(dataset):
    sp = {}
    sp['train_net'] = '"'+join(dirname(realpath(__file__)), "train.prototxt")+'"'
    sp['test_net'] = '"'+join(dirname(realpath(__file__)), "test_kld.prototxt")+'"'
    sp['base_lr'] = '1e2'
    sp['lr_policy'] = '"fixed"'
    sp['momentum'] = '0.9'
    #sp['stepsize'] = '100'
    sp['display'] = '100000'
    sp['snapshot'] = '10000' # never snapshot, we will snapshot manualy
    sp['snapshot_prefix'] = '"'+ temp_dir + dataset + '"'
    #sp['gamma'] = '0.5'
    sp['max_iter'] = '100000'
    sp['test_interval'] = '10000' # never test, we will test manualy
    sp['test_iter'] = '1'
    sp['debug_info'] = 'false'

    f = open(join(dirname(realpath(__file__)), 'solver.prototxt'), 'w')
    for k, v in sorted(sp.items()):
        if not(type(v) is str):
            raise TypeError('All solver parameters must be strings')
        f.write('%s: %s\n'%(k, v))
    f.close()

maxIterTable = dict({"Movie":2000})

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    datasets = [os.path.splitext(dataset)[0] for dataset in os.listdir('data/ldl/DataSets') if ('.mat' in dataset and '-shuffled' not in dataset and 'Human_Gene' not in dataset )]
    datasets.sort()
    f = open(join(dirname(realpath(__file__)), 'result.txt'), 'w')
    for dataset in datasets:
        record = open(join(dirname(realpath(__file__)), dataset+'-record.txt'), 'w')
        best_kld = np.zeros((10,2),dtype=np.float)
        fig = plt.figure();fig.hold(True);fig.suptitle(dataset+': KLD vs Iter');plt.xlabel('iter times');plt.ylabel('KLD')
        for idx in range(10):
            record.write("Cross Validation %d-----------------------------------------------------------\n"%(idx+1))
            make_net(dataset, idx)
            make_solver(dataset)
            solver = caffe.SGDSolver(join(dirname(realpath(__file__)), 'solver.prototxt'))
            current_best_kld = np.float32(100)
            if dataset in maxIterTable.keys():
                maxIter = maxIterTable[dataset]
            else:
                maxIter = 3000
            kld = np.zeros([maxIter,])
            for iter in range(maxIter):
                solver.step(1)
                # test net
                solver.test_nets[0].share_with(solver.net)
                kld[iter] = solver.test_nets[0].forward()['KLD']
                pred = solver.test_nets[0].blobs['pred'].data[...].copy()
                label = solver.test_nets[0].blobs['label'].data[...].copy()
                # save intermediate testing results for debugging
                if kld[iter] < current_best_kld:
                    current_best_kld = kld[iter]
                    io.savemat(join(temp_dir, "%s-%d-iter%d-label-pred.mat"%(dataset, idx, iter)), {'label':label, 'pred':pred, 'kld':kld[iter]})
                    # snapshot caffemodel
                    solver.test_nets[0].save(join(temp_dir, "%s-%d-iter%d-kld%f.caffemodel"%(dataset, idx, iter, kld[iter])))
                record.write("%s iter:%d KLD=%f BestKLD=%f \n"%(dataset, iter, kld[iter], current_best_kld))
                print "dataset: %s-%d, KLD=%f, best_kld=%f, iteration %d of %d"%(dataset, idx, kld[iter], current_best_kld,iter+1, maxIter)
            plt.plot(kld, label='cross-val:'+str(idx))
            best_kld[idx, 0] = current_best_kld
            best_kld[idx, 1] = np.float(iter)
        record.close()
        plt.legend(loc='best');
        fig.savefig(join(dirname(realpath(__file__)), dataset+'-kld.png'))
        fig.savefig(join(dirname(realpath(__file__)), dataset+'-kld.eps'), format='eps', dpi=1000)
        f.write('%s-kld:%f +- %f\n'%(dataset, np.mean(best_kld[:, 0]), np.std(best_kld[:, 0])))
        io.savemat(join(dirname(realpath(__file__)), "%s-kld-VS-iter-times.mat"%(dataset)), {'kld':best_kld})
    f.close()
