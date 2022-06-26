#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn_v8.h>

#define BLOCKDIM_X 256
#define CONV_FWD_ALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
#define CONV_BWD_FILTER_ALGO CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
#define CONV_BWD_DATA_ALGO CUDNN_CONVOLUTION_BWD_DATA_ALGO_1

typedef struct _TensorW {
    uint length;
    float *_tensor_;
    float *_gradient_;
    float *momentum_v;
} TensorW;

typedef struct _Node {
    void *_content_;
    cudnnTensorDescriptor_t yDesc;
    float *y;
    float *dy;
    uint N, C, H, W;
} Node;

typedef struct _Convolution {
    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t fDesc;
    cudnnTensorDescriptor_t bDesc;
    TensorW filter;
    TensorW bias;
} Convolution;

typedef struct _FullyConnected {
    TensorW W;
    TensorW b;
    uint in, out;
} FullyConnected;

typedef struct _Context {
    cublasHandle_t cublas;
    cudnnHandle_t cudnn;
    cudaStream_t stream;
    void *work_space;
    size_t space_size;
    TensorW **W_list;
    uint W_num;
} Context;

void exitWithInfo(uint line, const char *src_name);
void cuda(cudaError_t error_id, uint line, const char *src_name);
void cublas(cublasStatus_t status, uint line, const char *src_name);
void cudnn(cudnnStatus_t status, uint line, const char *src_name);
void GEMM(cublasHandle_t handle, int mode, const float *d_A, const float *d_B, float *d_Z,
          float alpha, float beta, uint AH, uint AW_BH, uint BW);
// op == 1: Y[i] = A[i] * B[i]
// op == 2: Y[i] = A[i] + B[i]
// op == 3: Y[i] += A[i] * B[i]
// op == 4: Y[i] = A[i]
void tensorOption(cudaStream_t stream, int op, float *Y, float *A, float *B, uint length);
// op == 1: Y[i] = A[i] * B[i]
// op == 2: Y[i] = A[i] + B[i]
// op == 3: Y[i] += A[i] * B[i]
// op == 4: Y[i] = A[i]
void tensorOptionBlocking(int op, float *Y, float *A, float *B, uint length);
void tensorScale(float *tensor, float value, uint length);

void momentunTensorW(TensorW *tensor, float m, float lr);

void initContext(Context *context);
void freeContext(Context *context);
void setStream(Context *context, cudaStream_t stream);
void addToWeightList(Context *context, TensorW *tensor);
void setGradientToZero(Context *context);
void momentun(Context *context, float m, float lr);

void initNode(Node *node, uint N, uint C, uint H, uint W);
void freeNode(Node *node);

void initConvolution2d(Context *context, Node *node, uint KN, uint C, uint KH, uint KW,
                       uint N, uint YH, uint YW, int stride, int pad);
void freeConvolution2d(Node *node);
void forwardConvolution2d(Context *context, Node *node, Node *x);
void backwardConvolution2d(Context *context, Node *node, Node *x);

void forwardUpsampling2x(cudaStream_t stream, Node *node, Node *x);
void backwardUpsampling2x(cudaStream_t stream, Node *node, Node *x);

void initMaxpooling2d2x2s2p0(Node *node, uint N, uint C, uint YH, uint YW);
void freePooling(Node *node);
void forwardPooling(Context *context, Node *node, Node *x);
void backwardPooling(Context *context, Node *node, Node *x);

void forwardChannelPooling(cudaStream_t stream, Node *node, Node *x);
void backwardChannelPooling(cudaStream_t stream, Node *node, Node *x);

void initFullyConnected(Context *context, Node *node, uint N, uint in, uint out);
void initFullyConnectedReshape(Context *context, Node *node, uint N, uint in,
                               uint out_C, uint out_H, uint out_W);
void freeFullyConnected(Node *node);
void forwardFullyConnected(Context *context, Node *node, Node *x);
void backwardFullyConnected(Context *context, Node *node, Node *x);

void forwardTanh(cudaStream_t stream, Node *node, Node *x);
void backwardTanh(cudaStream_t stream, Node *node, Node *x);
void forwardSigmoid(cudaStream_t stream, Node *node, Node *x);
void backwardSigmoid(cudaStream_t stream, Node *node, Node *x);
void forwardIPLU(cudaStream_t stream, Node *node, Node *x);
void backwardIPLU(cudaStream_t stream, Node *node, Node *x);

void printTensorDescriptor(cudnnTensorDescriptor_t desc);
void printFilterDescriptor(cudnnFilterDescriptor_t fDesc);
void printConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc);

// typedef struct _sfEncoder {
//     // 3 * 192 * 256
//     Node conv1; // 16 * 192 * 256 (5x5)
//     Node iplu1; // 16 * 192 * 256
//     Node pool1; // 16 * 96 * 128
//     Node conv2; // 32 * 96 * 128 (5x5)
//     Node iplu2; // 32 * 96 * 128
//     Node pool2; // 32 * 48 * 64
//     Node conv3; // 64 * 48 * 64 (5x5)
//     Node iplu3; // 64 * 48 * 64
//     Node pool3; // 64 * 24 * 32
//     Node conv4; // 128 * 24 * 32 (3x3)
//     Node iplu4; // 128 * 24 * 32
//     Node pool4; // 128 * 12 * 16
//     Node conv5; // 256 * 12 * 16 (3x3)
//     Node iplu5; // 256 * 12 * 16
//     Node pool5; // 256 * 6 * 8
//     Node conv6; // 512 * 6 * 8 (3x3)
//     Node iplu6; // 512 * 6 * 8
//     Node pool6; // 512 * 3 * 4
// } sfEncoder;

// typedef struct _sfGRUh {
//     Node h; // 1024
//     Node fc1; // 1024
//     Node iplu; // 1024
//     Node fc2; // 1024
// } sfGRUh;

// typedef struct _sfGRUx {
//     Node x; // 512 * 3 * 4
//     Node fc1; // 1024
//     Node iplu; // 1024
//     Node fc2; // 1024
// } sfGRUx;

// typedef struct _SoulFlow {
//     Context context;
//     cudaStream_t *streams;
//     cudaEvent_t event;
//     uint batch_size;

//     float *images_host;
//     Node images; // 3 * 192 * 256
//     sfEncoder encoder; // 512 * 3 * 4
//     Node img_fc1; // 1024
//     Node img_iplu; // 1024
//     Node img_fc2; // 7

//     sfGRUh gru_hr;
//     sfGRUx gru_xr;
//     Node sigmoid_r;
//     sfGRUh gru_hz;
//     sfGRUx gru_xz;
//     Node sigmoid_z;
//     Node h2;
//     sfGRUh gru_hh2;
//     sfGRUx gru_xh2;
//     Node tanh_h2;

//     float **x;
//     float **h;
//     uint h_id;
//     uint h_size;

//     Node ht; // 1024
//     Node out_fc1; // 1024
//     Node out_iplu; // 1024
//     Node out_fc2; // 7
// } SoulFlow;
