import numpy as np
import ctypes
import psutil

class tensor:
    def __init__(self):
        from lib import module
        self.module = module
        self.openblas = ctypes.cdll.LoadLibrary("./lib/libopenblas.dll")
        self.openblas.cblas_dgemv.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        self.openblas.cblas_dgemv.restype = None
        self.openblas.openblas_set_num_threads.argtypes = [ctypes.c_int]
        self.openblas.openblas_set_num_threads.restype = None
        self.openblas.openblas_set_num_threads(psutil.cpu_count(logical=False))

    def dot(self, A, B):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        K = A.size
        K2, N = B.shape
        C = np.zeros(N, dtype=np.float64)
        self.openblas.cblas_dgemv(
            101,
            112,
            K2,
            N,
            1.0,
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            N,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            1,
            0.0,
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            1
        )
        
        return C

    def sum(self, A, B):
        return self.module.tensor.sum(A, B)

class optimized:
    def __init__(self):
        self.optimized = ctypes.CDLL("./lib/optimized.dll")
        self.optimized.ReLU.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.optimized.ReLU.restype = None
        self.optimized.softmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.optimized.softmax.restype = None
        self.optimized.loss.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.optimized.loss.restype = ctypes.c_float
        self.optimized.regularize.argtypes = [ctypes.c_double, ctypes.c_double]
        self.optimized.regularize.restype = ctypes.c_double
        self.optimized.update.restype = ctypes.c_double
        self.optimized.update.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double
        ]

    def float_pointer(self, obj):
        return obj.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def double_pointer(self, obj):
        return obj.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def array_pointer(self, mats):
        arr = (ctypes.POINTER(ctypes.c_double) * len(mats))()
        for i, m in enumerate(mats):
            arr[i] = self.double_pointer(m)
        return arr
        
    def update(self, w, b, dw, db, layers_ptr, layers_num, lr, ld):
        w_ptr  = self.array_pointer([w[i].ravel() for i in range(layers_num)])
        dw_ptr = self.array_pointer([dw[i].ravel() for i in range(layers_num)])
        b_ptr  = self.array_pointer(b)
        db_ptr = self.array_pointer(db)

        w_mean = self.optimized.update(
            w_ptr, dw_ptr,
            b_ptr, db_ptr,
            layers_ptr,
            layers_num,
            lr,
            ld
        )
        return w_mean

    def regularize(self, w_mean, ld):
        return self.optimized.regularize(w_mean, ld)

    def ReLU(self, inp):
        if inp.dtype != np.float32:
            inp = inp.astype(np.float32)
        self.optimized.ReLU(self.float_pointer(inp), inp.size)
        return inp

    def softmax(self, inp):
        if inp.dtype != np.float32:
            inp = inp.astype(np.float32)
        out = np.zeros_like(inp, dtype=np.float32)
        self.optimized.softmax(self.float_pointer(inp), self.float_pointer(out), inp.size)
        return out

    def loss(self, yhat, ytrue):
        return self.optimized.loss(self.float_pointer(yhat), self.float_pointer(ytrue), ytrue.size)