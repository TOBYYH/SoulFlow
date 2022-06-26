import numpy as np
import ctypes
import os
import pickle
# import matplotlib.pylab as plt


class SoulFlow:
    def __init__(self, batch_size:int, path:str=None) -> None:
        self.c_lib = ctypes.cdll.LoadLibrary(os.getcwd() + "/libsoulflow.so")
        self.c_lib.initSoulFlow.argtypes = (ctypes.c_uint, )
        self.c_lib.initSoulFlow.restype = ctypes.c_void_p
        self.c_lib.freeSoulFlow.argtypes = (ctypes.c_void_p, )
        self.c_lib.sfGetWeightNum.argtypes = (ctypes.c_void_p, )
        self.c_lib.sfGetWeightNum.restype = ctypes.c_uint
        self.c_lib.sfGetWeightLength.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        self.c_lib.sfGetWeightLength.restype = ctypes.c_uint
        self.c_lib.sfCopyWeight.argtypes = (ctypes.c_void_p, ctypes.c_uint,
                                            ctypes.c_void_p, ctypes.c_int)
        self.c_lib.sfSetGradientToZero.argtypes = (ctypes.c_void_p, )
        self.c_lib.sfSetSample.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p)
        self.c_lib.forwardImage.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.c_lib.backwardImage.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.c_lib.momentunImage.argtypes = (ctypes.c_void_p, ctypes.c_float, ctypes.c_float)
        self.c_lib.forwardGRU.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.c_lib.predictGRU.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.c_lib.backwardGRU.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.c_lib.momentunGRU.argtypes = (ctypes.c_void_p, ctypes.c_float, ctypes.c_float)
        self.c_lib.resetTemp.argtypes = (ctypes.c_void_p, )
        self.c_lib.sfHookTensor.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.c_lib.sfHookTensor.restype = ctypes.c_uint

        self.c_lib.cudaCheckDeviceMemUsage()
        self.context = self.c_lib.initSoulFlow(batch_size)
        self.c_lib.cudaCheckDeviceMemUsage()
        self.N = batch_size

        if path is not None:
            with open(path, 'rb') as f:
                self.W_list = pickle.load(f)
            self.copyWeight(1)
            return

        std = 0.04
        self.W_list = []
        W_num = self.c_lib.sfGetWeightNum(self.context)
        for i in range(W_num):
            length = self.c_lib.sfGetWeightLength(self.context, i)
            w = np.float32(np.random.randn(length) * std)
            self.W_list.append(w)
        self.copyWeight(1)
    
    def copyWeight(self, mode:int):
        W_num = self.c_lib.sfGetWeightNum(self.context)
        for i in range(W_num):
            self.c_lib.sfCopyWeight(self.context, i, self.W_list[i].ctypes.data, mode)
    
    def deviceMemUsage(self):
        self.c_lib.cudaCheckDeviceMemUsage()
    
    # def show_histogram(self):
    #     length = self.c_lib.sfHookTensor(self.context, None)
    #     data = np.zeros([length], dtype=np.float32)
    #     self.c_lib.sfHookTensor(self.context, data.ctypes.data)
    #     print("max:", data.max())
    #     print("min:", data.min())
    #     plt.hist(data, 1000, range=(data.min(), data.max()))
    #     plt.show()
    
    def set_sample(self, N_id:int, image:np.ndarray):
        if image.shape[0] != 192:
            print("set_sample: Shape error:", image.shape)
            return
        if image.shape[1] != 256:
            print("set_sample: Shape error:", image.shape)
            return
        if image.shape[2] != 3:
            print("set_sample: Shape error:", image.shape)
            return
        self.c_lib.sfSetSample(self.context, N_id, image.ctypes.data)
    
    def forward_image(self):
        result = np.zeros([self.N, 7], dtype=np.float32)
        self.c_lib.forwardImage(self.context, result.ctypes.data)
        exp = np.exp(result)
        return exp / np.sum(exp, axis=1).reshape([self.N, 1])
    
    def backward_image(self, gradient:np.ndarray):
        self.c_lib.backwardImage(self.context, gradient.ctypes.data)
    
    def momentum_image(self, m, lr):
        self.c_lib.momentunImage(self.context, m, lr)
    
    def forward_gru(self, train:int):
        self.c_lib.forwardGRU(self.context, train)
    
    def predict_gru(self):
        result = np.zeros([self.N, 7], dtype=np.float32)
        self.c_lib.predictGRU(self.context, result.ctypes.data)
        exp = np.exp(result)
        return exp / np.sum(exp, axis=1).reshape([self.N, 1])
    
    def backward_gru(self, gradient:np.ndarray):
        self.c_lib.backwardGRU(self.context, gradient.ctypes.data)
    
    def momentum_gru(self, m, lr):
        self.c_lib.momentunGRU(self.context, m, lr)
    
    def setGradientToZero(self):
        self.c_lib.sfSetGradientToZero(self.context)
    
    def resetTemp(self):
        self.c_lib.resetTemp(self.context)
    
    def deviceSync(self):
        self.c_lib.deviceSynchronize()
    
    def free(self):
        self.c_lib.freeSoulFlow(self.context)
        self.c_lib.cudaCheckDeviceMemUsage()
    
    def to_storage(self, path:str):
        self.copyWeight(0)
        with open(path, 'wb') as f:
            pickle.dump(self.W_list, f, -1)
