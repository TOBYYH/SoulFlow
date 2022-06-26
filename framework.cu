#include "framework.h"

const char framework[] = "framework.cu";

void exitWithInfo(uint line, const char *src_name)
{
    printf("Exception in %s at line %d.\n", src_name, line);
    exit(EXIT_FAILURE);
}

void cuda(cudaError_t error_id, uint line, const char *src_name)
{
    if (error_id != cudaSuccess) {
        printf("Error: CUDA function returned %d at line %d\n    -> %s\n",
            (int)error_id, line, cudaGetErrorString(error_id));
        printf("    -> source file: %s\n", src_name);
        exit(EXIT_FAILURE);
    }
}

void cublas(cublasStatus_t status, uint line, const char *src_name)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Error: cuBLAS function returned %d at line %d\n    -> %s\n",
            (int)status, line, cublasGetStatusString(status));
        printf("    -> source file: %s\n", src_name);
        exit(EXIT_FAILURE);
    }
}

void cudnn(cudnnStatus_t status, uint line, const char *src_name)
{
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("Error: cuDNN function returned %d at line %d\n    -> %s\n",
            (int)status, line, cudnnGetErrorString(status));
        printf("    -> source file: %s\n", src_name);
        exit(EXIT_FAILURE);
    }
}

extern "C"
void cuda_info(void) {
    int deviceCount = 0;
    cuda(cudaGetDeviceCount(&deviceCount), __LINE__, framework);
    cuda(cudaGetLastError(), __LINE__, framework);
    if (deviceCount != 0)
        printf("==========Detected %d CUDA Capable device(s)==========\n", deviceCount);
    else printf("There are no available device(s) that support CUDA\n");
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    while(dev < deviceCount) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("    CUDA Driver Version / Runtime Version:          %d.%d / %d.%d\n",
            driverVersion/1000, (driverVersion%100)/10,
            runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("    CUDA Capability Major/Minor version number:     %d.%d\n",
            deviceProp.major, deviceProp.minor);
        printf("    Total amount of global memory:                  %.3f GBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem/pow(1024.0, 3),
            (unsigned long long)deviceProp.totalGlobalMem);
        printf("    GPU Clock rate:                                 %.1f MHz\n",
            deviceProp.clockRate * 1e-3f);
        printf("    Memory Clock rate:                              %.1f MHz\n",
            deviceProp.memoryClockRate * 1e-3f);
        printf("    Memory Bus Width:                               %d-bit\n",
            deviceProp.memoryBusWidth);
        if (deviceProp.l2CacheSize)
            printf("    L2 Cache Size:                                  %d bytes\n",
                deviceProp.l2CacheSize);
        printf("    Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("    Max Layered Texture Size (dim) * layers 1D=(%d) * %d, 2D=(%d,%d) * %d\n",
            deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);
        printf("    Total amount of constant memory:                %lu bytes\n",
            deviceProp.totalConstMem);
        printf("    Total amount of shared memory per block:        %lu bytes\n",
            deviceProp.sharedMemPerBlock);
        printf("    Total number of registers available per block:  %d\n",
            deviceProp.regsPerBlock);
        printf("    Warp size:                                      %d\n", deviceProp.warpSize);
        printf("    Number of multiprocessor:                       %d\n", deviceProp.multiProcessorCount);
        printf("    Maxinum number of threads per multiprocessor:   %d\n",
            deviceProp.maxThreadsPerMultiProcessor);
        printf("    Maxinum number of threads per block:            %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("    Maxinum sizes of each dimension of a block:     %d * %d * %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("    Maxinum sizes of each dimension of a grid:      %d * %d * %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("    Maxinum memory pitch:                           %lu bytes\n", deviceProp.memPitch);
        printf("    Shared memory bank size:                        ");
        cudaSharedMemConfig smConfig;
        cudaDeviceGetSharedMemConfig(&smConfig);
        if (smConfig == cudaSharedMemBankSizeFourByte) printf("%d\n", 4);
        else if (smConfig == cudaSharedMemBankSizeEightByte) printf("%d\n", 8);
        cudaDeviceReset();
        dev += 1;
    }
}

extern "C"
void cudaCheckDeviceMemUsage(void)
{
    size_t total, free;
    cuda(cudaMemGetInfo(&free, &total), __LINE__, framework);
    printf("cudaCheckDeviceMemUsage: Total: %.3f MB, used: %.3f MB, free: %.3f MB\n",
           (float)(total/pow(1024.0f, 2.0f)),
           (float)((total-free)/pow(1024.0f, 2.0f)),
           (float)(free/pow(1024.0f, 2.0f)));
}

// mode == 1: transpose A
// mode == 2: transpose B
// mode == 3: transpose A & B
// else: no transpose
void GEMM(cublasHandle_t handle, int mode, const float *d_A, const float *d_B, float *d_Z,
          float alpha, float beta, uint AH, uint AW_BH, uint BW)
{
    cublasOperation_t transa, transb;
    switch (mode) {
        case 1:
        transa = CUBLAS_OP_N;
        transb = CUBLAS_OP_T;
        cublas(cublasSgemm_v2(handle, transa, transb,
                              BW, AH, AW_BH,
                              &alpha,
                              d_B, BW,
                              d_A, AH,
                              &beta,
                              d_Z, BW), __LINE__, framework);
        return;
        case 2:
        transa = CUBLAS_OP_T;
        transb = CUBLAS_OP_N;
        cublas(cublasSgemm_v2(handle, transa, transb,
                              BW, AH, AW_BH,
                              &alpha,
                              d_B, AW_BH,
                              d_A, AW_BH,
                              &beta,
                              d_Z, BW), __LINE__, framework);
        return;
        case 3:
        transa = CUBLAS_OP_T;
        transb = CUBLAS_OP_T;
        cublas(cublasSgemm_v2(handle, transa, transb,
                              BW, AH, AW_BH,
                              &alpha,
                              d_B, AW_BH,
                              d_A, BW,
                              &beta,
                              d_Z, BW), __LINE__, framework);
        return;
        default:
        transa = CUBLAS_OP_N;
        transb = CUBLAS_OP_N;
        cublas(cublasSgemm_v2(handle, transa, transb,
                              BW, AH, AW_BH,
                              &alpha,
                              d_B, BW,
                              d_A, AW_BH,
                              &beta,
                              d_Z, BW), __LINE__, framework);
    }
}

__global__ void deviceMemSet_K(float *dst, uint length, float value)
{
    uint index = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (index >= length) return;
    dst[index] = value;
}

void deviceMemSet(cudaStream_t stream, float *dst, uint length, float value)
{
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    deviceMemSet_K<<<grid, block, 0, stream>>>(dst, length, value);
}

void deviceMemSetBlocking(float *dst, uint length, float value)
{
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    deviceMemSet_K<<<grid, block>>>(dst, length, value);
}

// tensorOption====================================================================================
__global__ void tensorMultiply_K(float *Y, float *A, float *B, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    Y[id] = A[id] * B[id];
}

__global__ void tensorAdd_K(float *Y, float *A, float *B, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    Y[id] = A[id] + B[id];
}

__global__ void tensorMultiplyAdd_K(float *Y, float *A, float *B, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    Y[id] += A[id] * B[id];
}

__global__ void tensorCopy_K(float *Y, float *A, float *B, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    Y[id] = A[id];
}

void tensorOption(cudaStream_t stream, int op, float *Y, float *A, float *B, uint length)
{
    // op == 1: Y[i] = A[i] * B[i]
    // op == 2: Y[i] = A[i] + B[i]
    // op == 3: Y[i] += A[i] * B[i]
    // op == 4: Y[i] = A[i]
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    switch (op) {
        case 1:
        tensorMultiply_K<<<grid, block, 0, stream>>>(Y, A, B, length); break;
        case 2:
        tensorAdd_K<<<grid, block, 0, stream>>>(Y, A, B, length); break;
        case 3:
        tensorMultiplyAdd_K<<<grid, block, 0, stream>>>(Y, A, B, length); break;
        case 4:
        tensorCopy_K<<<grid, block, 0, stream>>>(Y, A, B, length); break;
        default:
        printf("tensorOption: Invalid value of op.\n");
        exit(-1);
    }
}

void tensorOptionBlocking(int op, float *Y, float *A, float *B, uint length)
{
    // op == 1: Y[i] = A[i] * B[i]
    // op == 2: Y[i] = A[i] + B[i]
    // op == 3: Y[i] += A[i] * B[i]
    // op == 4: Y[i] = A[i]
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    switch (op) {
        case 1:
        tensorMultiply_K<<<grid, block>>>(Y, A, B, length); break;
        case 2:
        tensorAdd_K<<<grid, block>>>(Y, A, B, length); break;
        case 3:
        tensorMultiplyAdd_K<<<grid, block>>>(Y, A, B, length); break;
        case 4:
        tensorCopy_K<<<grid, block>>>(Y, A, B, length); break;
        default:
        printf("tensorOptionBlocking: Invalid value of op.\n");
        exit(-1);
    }
}

__global__ void tensorScale_K(float *tensor, float value, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    tensor[id] = tensor[id] * value;
}

void tensorScale(float *tensor, float value, uint length)
{
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    tensorScale_K<<<grid, block>>>(tensor, value, length);
}

// TensorW=========================================================================================
void initTensorW(TensorW *tensor, uint length)
{
    tensor->length = length;
    size_t size = sizeof(float) * length;
    cuda(cudaMalloc(&(tensor->_tensor_), size), __LINE__, framework);
    cuda(cudaMalloc(&(tensor->_gradient_), size), __LINE__, framework);
    cuda(cudaMalloc(&(tensor->momentum_v), size), __LINE__, framework);
    cuda(cudaMemset(tensor->momentum_v, 0, size), __LINE__, framework);
}

void freeTensorW(TensorW *tensor)
{
    // printf("freeTensorW: %p\n", tensor->_tensor_);
    cuda(cudaFree(tensor->_tensor_), __LINE__, framework);
    cuda(cudaFree(tensor->_gradient_), __LINE__, framework);
    cuda(cudaFree(tensor->momentum_v), __LINE__, framework);
}

#define WEIGHT_DECAY 0.1f

__global__ void momentunTensorW_K(float *_tensor_, float *_gradient_, float *momentum_v,
                                  float m, float lr, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    float w = _tensor_[id];
    float gradient = _gradient_[id] + w * WEIGHT_DECAY;
    if (gradient > 1.0f) gradient = 1.0f;
    if (gradient < -1.0f) gradient = -1.0f;
    float v = momentum_v[id] * m - gradient * lr;
    momentum_v[id] = v;
    _tensor_[id] = w + v;
}

void momentunTensorW(TensorW *tensor, float m, float lr)
{
    uint gridx = tensor->length / BLOCKDIM_X;
    if (tensor->length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    momentunTensorW_K<<<grid, block>>>(tensor->_tensor_, tensor->_gradient_,
                                       tensor->momentum_v, m, lr, tensor->length);
}

// Context=========================================================================================
void initContext(Context *context)
{
    cublas(cublasCreate(&(context->cublas)), __LINE__, framework);
    cudnn(cudnnCreate(&(context->cudnn)), __LINE__, framework);
    context->work_space = NULL;
    context->space_size = 0;
    context->W_list = NULL;
    context->W_num = 0;
}

void freeContext(Context *context)
{
    cublas(cublasDestroy(context->cublas), __LINE__, framework);
    cudnn(cudnnDestroy(context->cudnn), __LINE__, framework);
    if (context->W_list != NULL) free(context->W_list);
}

void setStream(Context *context, cudaStream_t stream)
{
    cudnn(cudnnSetStream(context->cudnn, stream), __LINE__, framework);
    cublas(cublasSetStream(context->cublas, stream), __LINE__, framework);
    context->stream = stream;
}

void addToWeightList(Context *context, TensorW *tensor)
{
    if (context->W_num % 10 == 0) {
        size_t size = sizeof(void *) * (context->W_num + 10);
        context->W_list = (TensorW **)realloc(context->W_list, size);
    }
    context->W_list[context->W_num] = tensor;
    context->W_num += 1;
}

void setGradientToZero(Context *context)
{
    for (uint i = 0; i < context->W_num; i++) {
        TensorW *tensor = context->W_list[i];
        size_t size = sizeof(float) * tensor->length;
        cuda(cudaMemset(tensor->_gradient_, 0, size), __LINE__, framework);
    }
}

void momentun(Context *context, float m, float lr)
{
    for (uint i = 0; i < context->W_num; i++)
        momentunTensorW(context->W_list[i], m, lr);
    cuda(cudaDeviceSynchronize(), __LINE__, framework);
}

// Node============================================================================================
void initNode(Node *node, uint N, uint C, uint H, uint W)
{
    node->_content_ = NULL;
    cudnn(cudnnCreateTensorDescriptor(&(node->yDesc)), __LINE__, framework);
    cudnn(cudnnSetTensor4dDescriptor(node->yDesc,
                                     CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                     N, C, H, W), __LINE__, framework);
    size_t size = sizeof(float) * N * C * H * W;
    cuda(cudaMalloc(&(node->y), size), __LINE__, framework);
    cuda(cudaMalloc(&(node->dy), size), __LINE__, framework);
    node->N = N;
    node->C = C;
    node->H = H;
    node->W = W;
}

void freeNode(Node *node)
{
    cudnn(cudnnDestroyTensorDescriptor(node->yDesc), __LINE__, framework);
    cuda(cudaFree(node->y), __LINE__, framework);
    cuda(cudaFree(node->dy), __LINE__, framework);
}

// Convolution=====================================================================================
void initConvolution2d(Context *context, Node *node, uint KN, uint C, uint KH, uint KW,
                       uint N, uint YH, uint YW, int stride, int pad)
{
    initNode(node, N, KN, YH, YW);
    Convolution *conv = (Convolution *)malloc(sizeof(Convolution));
    cudnn(cudnnCreateConvolutionDescriptor(&(conv->convDesc)), __LINE__, framework);
    cudnn(cudnnSetConvolution2dDescriptor(conv->convDesc,
                                          pad, pad, stride, stride, 1, 1, CUDNN_CONVOLUTION,
                                          CUDNN_DATA_FLOAT), __LINE__, framework);
    cudnn(cudnnSetConvolutionMathType(conv->convDesc,
                                      CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION),
                                     __LINE__, framework);
    cudnn(cudnnCreateFilterDescriptor(&(conv->fDesc)), __LINE__, framework);
    cudnn(cudnnSetFilter4dDescriptor(conv->fDesc,
                                     CUDNN_DATA_FLOAT,
                                     CUDNN_TENSOR_NCHW,
                                     KN, C, KH, KW), __LINE__, framework);
    cudnn(cudnnCreateTensorDescriptor(&(conv->bDesc)), __LINE__, framework);
    cudnn(cudnnSetTensor4dDescriptor(conv->bDesc,
                                     CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                     1, KN, 1, 1), __LINE__, framework);
    initTensorW(&(conv->filter), KN * C * KH * KW);
    initTensorW(&(conv->bias), KN);
    addToWeightList(context, &(conv->filter));
    addToWeightList(context, &(conv->bias));
    node->_content_ = (void *)conv;

    int H = (YH - 1) * stride + KH - pad * 2;
    int W = (YW - 1) * stride + KW - pad * 2;
    cudnnTensorDescriptor_t tempDesc;
    cudnn(cudnnCreateTensorDescriptor(&tempDesc), __LINE__, framework);
    cudnn(cudnnSetTensor4dDescriptor(tempDesc,
                                     CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                     N, C, H, W), __LINE__, framework);
    size_t temp_size;
    cudnn(cudnnGetConvolutionForwardWorkspaceSize(context->cudnn, tempDesc,
                                                  conv->fDesc, conv->convDesc,
                                                  node->yDesc, CONV_FWD_ALGO,
                                                  &temp_size), __LINE__, framework);
    if (temp_size > context->space_size) context->space_size = temp_size;
    cudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(context->cudnn, tempDesc,
                                                         node->yDesc, conv->convDesc,
                                                         conv->fDesc, CONV_BWD_FILTER_ALGO,
                                                         &temp_size), __LINE__, framework);
    if (temp_size > context->space_size) context->space_size = temp_size;
    cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(context->cudnn, conv->fDesc,
                                                       node->yDesc, conv->convDesc,
                                                       tempDesc, CONV_BWD_DATA_ALGO,
                                                       &temp_size), __LINE__, framework);
    if (temp_size > context->space_size) context->space_size = temp_size;
    cudnn(cudnnDestroyTensorDescriptor(tempDesc), __LINE__, framework);
}

void freeConvolution2d(Node *node)
{
    if (node->_content_ == NULL) exitWithInfo(__LINE__, framework);
    Convolution *conv = (Convolution *)node->_content_;
    cudnn(cudnnDestroyConvolutionDescriptor(conv->convDesc), __LINE__, framework);
    cudnn(cudnnDestroyFilterDescriptor(conv->fDesc), __LINE__, framework);
    cudnn(cudnnDestroyTensorDescriptor(conv->bDesc), __LINE__, framework);
    freeTensorW(&(conv->filter));
    freeTensorW(&(conv->bias));
    free(conv);
    freeNode(node);
}

__global__ void forwardConvolution2dBias_K(float *y, float *b, uint yc_size, uint yn_size, uint N)
{
    __shared__ float bias;
    uint t_x = threadIdx.x;
    uint YC_id = blockIdx.y;
    if (t_x == 0) bias = b[YC_id];
    __syncthreads();
    uint YC_offset = BLOCKDIM_X * blockIdx.x + t_x;
    if (YC_offset >= yc_size) return;
    uint N_offset = yc_size * YC_id + YC_offset;
    for (uint n = 0; n < N; n++)
        y[yn_size * n + N_offset] += bias;
}

void forwardConvolution2d(Context *context, Node *node, Node *x)
{
    if (node->_content_ == NULL) exitWithInfo(__LINE__, framework);
    float alpha = 1.0f, beta = 0.0f;
    Convolution *conv = (Convolution *)node->_content_;
    cudnn(cudnnConvolutionForward(context->cudnn, &alpha, x->yDesc, x->y,
                                  conv->fDesc, conv->filter._tensor_,
                                  conv->convDesc, CONV_FWD_ALGO,
                                  context->work_space, context->space_size,
                                  &beta, node->yDesc, node->y)
                                  , __LINE__, framework);
    uint yc_size = node->H * node->W;
    uint yn_size = yc_size * node->C;
    uint gridx = yc_size / BLOCKDIM_X;
    if (yc_size % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx, node->C);
    forwardConvolution2dBias_K
        <<<grid, block, 0, context->stream>>>(node->y, conv->bias._tensor_,
                                              yc_size, yn_size, node->N);
}

void backwardConvolution2d(Context *context, Node *node, Node *x)
{
    // printf("backwardConvolution2d: t\n");
    if (node->_content_ == NULL) exitWithInfo(__LINE__, framework);
    float alpha = 1.0f, beta = 1.0f;
    Convolution *conv = (Convolution *)node->_content_;
    cudnn(cudnnConvolutionBackwardFilter(context->cudnn, &alpha, x->yDesc, x->y,
                                         node->yDesc, node->dy,
                                         conv->convDesc, CONV_BWD_FILTER_ALGO,
                                         context->work_space, context->space_size,
                                         &beta, conv->fDesc, conv->filter._gradient_),
                                         __LINE__, framework);
    cudnn(cudnnConvolutionBackwardBias(context->cudnn, &alpha, node->yDesc, node->dy,
                                       &beta, conv->bDesc, conv->bias._gradient_),
                                       __LINE__, framework);
    beta = 0.0f;
    cudnn(cudnnConvolutionBackwardData(context->cudnn, &alpha,
                                       conv->fDesc, conv->filter._tensor_,
                                       node->yDesc, node->dy,
                                       conv->convDesc, CONV_BWD_DATA_ALGO,
                                       context->work_space, context->space_size,
                                       &beta, x->yDesc, x->dy), __LINE__, framework);
}

// Pooling======================================================================================
void initMaxpooling2d2x2s2p0(Node *node, uint N, uint C, uint YH, uint YW)
{
    initNode(node, N, C, YH, YW);
    cudnnPoolingDescriptor_t desc;
    cudnn(cudnnCreatePoolingDescriptor(&desc), __LINE__, framework);
    cudnn(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                      2, 2, 0, 0, 2, 2), __LINE__, framework);
    node->_content_ = (void *)desc;
}

void freePooling(Node *node)
{
    if (node->_content_ == NULL) exitWithInfo(__LINE__, framework);
    cudnnPoolingDescriptor_t desc = (cudnnPoolingDescriptor_t)node->_content_;
    cudnn(cudnnDestroyPoolingDescriptor(desc), __LINE__, framework);
    freeNode(node);
}

void forwardPooling(Context *context, Node *node, Node *x)
{
    if (node->_content_ == NULL) exitWithInfo(__LINE__, framework);
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingDescriptor_t desc = (cudnnPoolingDescriptor_t)node->_content_;
    cudnn(cudnnPoolingForward(context->cudnn, desc, &alpha, x->yDesc, x->y,
                              &beta, node->yDesc, node->y), __LINE__, framework);
}

void backwardPooling(Context *context, Node *node, Node *x)
{
    if (node->_content_ == NULL) exitWithInfo(__LINE__, framework);
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingDescriptor_t desc = (cudnnPoolingDescriptor_t)node->_content_;
    cudnn(cudnnPoolingBackward(context->cudnn, desc,
                               &alpha, node->yDesc, node->y,
                               node->yDesc, node->dy, x->yDesc, x->y,
                               &beta, x->yDesc, x->dy), __LINE__, framework);
}

// FullyConnected==================================================================================
void initFullyConnected(Context *context, Node *node, uint N, uint in, uint out)
{
    initNode(node, N, 1, 1, out);
    FullyConnected *fc = (FullyConnected *)malloc(sizeof(FullyConnected));
    initTensorW(&(fc->W), in * out);
    initTensorW(&(fc->b), out);
    addToWeightList(context, &(fc->W));
    addToWeightList(context, &(fc->b));
    fc->in = in;
    fc->out = out;
    node->_content_ = (void *)fc;
}

void initFullyConnectedReshape(Context *context, Node *node, uint N, uint in,
                               uint out_C, uint out_H, uint out_W)
{
    initNode(node, N, out_C, out_H, out_W);
    uint out = out_C * out_H * out_W;
    FullyConnected *fc = (FullyConnected *)malloc(sizeof(FullyConnected));
    initTensorW(&(fc->W), in * out);
    initTensorW(&(fc->b), out);
    addToWeightList(context, &(fc->W));
    addToWeightList(context, &(fc->b));
    fc->in = in;
    fc->out = out;
    node->_content_ = (void *)fc;
}

void freeFullyConnected(Node *node)
{
    FullyConnected *fc = (FullyConnected *)node->_content_;
    freeTensorW(&(fc->W));
    freeTensorW(&(fc->b));
    free(fc);
    freeNode(node);
}

__global__ void forwardFullyConnectedBias_K(float *y, float *b, uint N, uint out)
{
    uint out_id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (out_id >= out) return;
    float bias = b[out_id];
    for (uint n = 0; n < N; n++)
        y[out * n + out_id] += bias;
}

void forwardFullyConnected(Context *context, Node *node, Node *x)
{
    FullyConnected *fc = (FullyConnected *)node->_content_;
    uint in = x->C * x->H * x->W;
    // printf("forwardFullyConnected: %d, %d\n", in, fc->in);
    if (in != fc->in) exitWithInfo(__LINE__, framework);

    float alpha = 1.0f, beta = 0.0f;
    GEMM(context->cublas, 0, x->y, fc->W._tensor_, node->y, alpha, beta, x->N, in, fc->out);

    uint gridx = fc->out / BLOCKDIM_X;
    if (fc->out % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    forwardFullyConnectedBias_K
        <<<grid, block, 0, context->stream>>>(node->y, fc->b._tensor_, node->N, fc->out);
}

__global__ void backwardFullyConnectedBias_K(float *dy, float *db, uint N, uint out)
{
    uint out_id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (out_id >= out) return;
    float dbias = 0;
    for (uint n = 0; n < N; n++)
        dbias += dy[out * n + out_id];
    db[out_id] += dbias;
}

void backwardFullyConnected(Context *context, Node *node, Node *x)
{
    FullyConnected *fc = (FullyConnected *)node->_content_;
    uint in = x->C * x->H * x->W;
    if (in != fc->in) exitWithInfo(__LINE__, framework);

    float alpha = 1.0f, beta = 1.0f;
    GEMM(context->cublas, 1, x->y, node->dy, fc->W._gradient_, alpha, beta, in, node->N, fc->out);

    uint gridx = fc->out / BLOCKDIM_X;
    if (fc->out % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    backwardFullyConnectedBias_K
        <<<grid, block, 0, context->stream>>>(node->dy, fc->b._gradient_, node->N, fc->out);
    
    beta = 0.0f;
    GEMM(context->cublas, 2, node->dy, fc->W._tensor_, x->dy, alpha, beta, node->N, fc->out, in);
}

// ================================================================================================
void printTensorDescriptor(cudnnTensorDescriptor_t desc)
{
    cudnnDataType_t dtype;
    int n, c, h, w;
    int nStride, cStride, hStride, wStride;
    cudnn(cudnnGetTensor4dDescriptor(desc, &dtype, &n, &c, &h, &w,
                                     &nStride, &cStride, &hStride, &wStride), __LINE__, framework);
    printf("==========printTensorDescriptor==========\n");
    printf("N: %d, C: %d, H: %d, W: %d\n", n, c, h, w);
    printf("N_s: %d, C_s: %d, H_s: %d, W_s: %d\n", nStride, cStride, hStride, wStride);
}

void printFilterDescriptor(cudnnFilterDescriptor_t fDesc)
{
    cudnnDataType_t dtype;
    cudnnTensorFormat_t format;
    int k, c, h, w;
    cudnn(cudnnGetFilter4dDescriptor(fDesc, &dtype, &format, &k, &c, &h, &w), __LINE__, framework);
    printf("==========printFilterDescriptor==========\n");
    printf("KN: %d, C: %d, H: %d, W: %d\n", k, c, h, w);
}

void printConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc)
{
    cudnnConvolutionMode_t mode;
    cudnnDataType_t computeType;
    int pad_h, pad_w, u, v, dilation_h, dilation_w;
    cudnn(cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &u, &v,
                                          &dilation_h, &dilation_w, &mode, &computeType),
                                          __LINE__, framework);
    printf("==========printConvolution2dDescriptor==========\n");
    printf("pad_h: %d, pad_w: %d, u: %d, v: %d, ", pad_h, pad_w, u, v);
    printf("dilation_h: %d, dilation_w: %d\n", dilation_h, dilation_w);
}
