import caffe
import numpy as np
import scipy.io as io
from os.path import join, isfile
class LDLDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']
        params = eval(self.param_str)
        self.db_name = params['db_name'] 
        self.batch_size = params['batch_size']
        self.split_idx = params['split_idx']
        self.phase = params['phase']
        if params.has_key('sub_mean'):
          self.sub_mean = params['sub_mean']
        else:
          self.sub_mean = False
        assert(self.split_idx <= 9)
        if isfile(join('data/ldl/DataSets/',self.db_name+'-shuffled.mat')):
            mat = io.loadmat(join('data/ldl/DataSets/',self.db_name+'-shuffled.mat'))
        else:
            mat = io.loadmat(join('data/ldl/DataSets/',self.db_name+'.mat'))
            data = mat['features']
            label = mat['labels']
            shuffle_idx = np.random.choice(label.shape[0], label.shape[0])
            data = data[shuffle_idx, :]
            label = label[shuffle_idx, :]
            mat = dict({'features':data, 'labels':label})
            io.savemat(join('data/ldl/DataSets/',self.db_name+'-shuffled.mat'), mat)
        self.features = mat['features']
        self.labels = mat['labels']
        self.N, self.D1 = self.features.shape
        _, self.D2 = self.labels.shape
        self.N = int(np.floor(self.labels.shape[0]/10)*10)
        # discard extra samples
        self.features = self.features[0:self.N, :]
        self.labels = self.labels[0:self.N, :]
        Ntest = self.N / 10
        self.Ntrain = int(self.N - Ntest)
        if self.phase=='test':
            assert(self.batch_size == Ntest)
        train_test_filter = np.array([False] * self.N)
        train_test_filter[self.split_idx*Ntest:(self.split_idx+1)*Ntest] = True
        self.test_data = self.features[train_test_filter, :]
        self.test_label = self.labels[train_test_filter, :]
        self.train_data = self.features[np.logical_not(train_test_filter), :]
        self.train_label = self.labels[np.logical_not(train_test_filter), :]
        if self.sub_mean:
            print "Subtract mean ... "
            data_mean = np.mean(self.train_data, 0)
            self.train_data = self.train_data - np.tile(data_mean, [self.train_data.shape[0], 1])
            self.test_data = self.test_data - np.tile(data_mean, [self.test_data.shape[0], 1])
        top[0].reshape(self.batch_size,self.D1,1,1)
        top[1].reshape(self.batch_size,self.D2,1,1)

    def forward(self, bottom, top):
        if self.phase == 'train':
            rnd_select = np.random.choice(self.Ntrain, self.batch_size)
            top[0].data[:,:,0,0] = self.train_data[rnd_select, :]
            top[1].data[:,:,0,0] = self.train_label[rnd_select, :]
        elif self.phase == 'test':
            top[0].data[:,:,0,0] = self.test_data
            top[1].data[:,:,0,0] = self.test_label

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

