

import h5py
import os

class HDF5DatasetWriter:
    def __init__(self,dims,output_path,data_key="image",buf_size=1000):
        """
            初始化HDF5写入器。

            参数:
            - dims: 一个元组，定义数据集的形状，例如 (样本总数, 高, 宽, 通道数)
            - output_path: 输出HDF5文件的路径
            - data_key: 存储数据的数据集名称，默认为'images'
            - buf_size: 缓冲区大小，当累积的数据量达到该值时，会写入磁盘
        """
        if os.path.exists(output_path):
            raise ValueError('already exists!')
        print('output_path:',output_path)
        self.db = h5py.File(output_path,'w')
        self.data = self.db.create_dataset(data_key,dims,dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0],),dtype='int')

        self.bufSize = buf_size
        self.buffer = {"data":[],"labels":[]}
        self.idx = 0

    def add(self,rows,labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)
        if len(self.buffer['data'])>self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data':[],'labels':[]}
    def store_class_labels(self,class_labels):
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset('label_names',(len(class_labels)),dtype=dt)
        label_set[:] = class_labels
    def close(self):
        if len(self.buffer['data'])>0:
            self.flush()
        self.db.close()


