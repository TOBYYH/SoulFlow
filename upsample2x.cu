#include "framework.h"

const char src_name[] = "upsample2x.cu";

typedef struct info {
    uint H, W, YH, YW;
    uint xc_size, xn_size, yc_size, yn_size;
} _info_;

__global__ void forwardUpsampling2x_K(float *x, float *y, _info_ info)
{
    uint C_id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (C_id >= info.xc_size) return;
    uint H_id = C_id / info.W;
    uint W_id = C_id % info.W;
    uint X_id = info.xn_size * blockIdx.z + info.xc_size * blockIdx.y + C_id;
    uint Y_offset = info.yn_size * blockIdx.z
                    + info.yc_size * blockIdx.y + info.YH * H_id*2 + W_id*2;
    y[Y_offset] = x[X_id];
    y[Y_offset + 1] = 0.0f;
    y[Y_offset + info.YW] = 0.0f;
    y[Y_offset + info.YW + 1] = 0.0f;
}

void forwardUpsampling2x(cudaStream_t stream, Node *node, Node *x)
{
    if (x->C != node->C) exitWithInfo(__LINE__, src_name);
    if (x->H * 2 != node->H) exitWithInfo(__LINE__, src_name);
    if (x->W * 2 != node->W) exitWithInfo(__LINE__, src_name);

    _info_ info;
    info.H = x->H;
    info.W = x->W;
    info.YH = node->H;
    info.YW = node->W;
    info.xc_size = info.H * info.W;
    info.xn_size = info.xc_size * x->C;
    info.yc_size = info.YH * info.YW;
    info.yn_size = info.yc_size * node->C;

    uint gridx = info.xc_size / BLOCKDIM_X;
    if (info.xc_size % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx, x->C, x->N);
    forwardUpsampling2x_K<<<grid, block, 0, stream>>>(x->y, node->y, info);
}

__global__ void backwardUpsampling2x_K(float *dx, float *dy, _info_ info)
{
    uint C_id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (C_id >= info.xc_size) return;
    uint H_id = C_id / info.W;
    uint W_id = C_id % info.W;
    uint X_id = info.xn_size * blockIdx.z + info.xc_size * blockIdx.y + C_id;
    uint Y_offset = info.yn_size * blockIdx.z
                    + info.yc_size * blockIdx.y + info.YH * H_id*2 + W_id*2;
    dx[X_id] = dy[Y_offset];
}

void backwardUpsampling2x(cudaStream_t stream, Node *node, Node *x)
{
    if (x->C != node->C) exitWithInfo(__LINE__, src_name);
    if (x->H * 2 != node->H) exitWithInfo(__LINE__, src_name);
    if (x->W * 2 != node->W) exitWithInfo(__LINE__, src_name);

    _info_ info;
    info.H = x->H;
    info.W = x->W;
    info.YH = node->H;
    info.YW = node->W;
    info.xc_size = info.H * info.W;
    info.xn_size = info.xc_size * x->C;
    info.yc_size = info.YH * info.YW;
    info.yn_size = info.yc_size * node->C;

    uint gridx = info.xc_size / BLOCKDIM_X;
    if (info.xc_size % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx, x->C, x->N);
    backwardUpsampling2x_K<<<grid, block, 0, stream>>>(x->dy, node->dy, info);
}

extern "C"
void TEST_forwardUpsampling2x(void)
{
    uint xlen = 10, C = 4, N = 2;
    Node input;
    Node node;
    initNode(&input, N, C, xlen, xlen);
    initNode(&node, N, C, xlen*2, xlen*2);
    size_t x_size = sizeof(float) * xlen * xlen * C * N;
    size_t y_size = sizeof(float) * xlen * xlen * 4 * C * N;
    float *x = (float *)malloc(x_size);
    for (uint i = 0; i < xlen * xlen * C * N; i++) x[i] = 0.1f * i;
    cuda(cudaMemcpy(input.y, x, x_size, cudaMemcpyHostToDevice), __LINE__, src_name);
    cudaStream_t stream;
    cuda(cudaStreamCreate(&stream), __LINE__, src_name);
    forwardUpsampling2x(stream, &node, &input);
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
    float *y = (float *)malloc(y_size);
    cuda(cudaMemcpy(y, node.y, y_size, cudaMemcpyDeviceToHost), __LINE__, src_name);
    uint id = 0;
    for (uint n = 0; n < N; n++) {
        for (uint i = 0; i < C; i++) {
            for (uint j = 0; j < xlen * xlen * 4; j++) {
                if (id % (xlen*2) == 0) printf("\n");
                printf("%4g ", y[id]);
                id += 1;
            }
            printf("\n");
        }
        printf("\n");
    }
    freeNode(&input);
    freeNode(&node);
    free(x);
    free(y);
    cuda(cudaStreamDestroy(stream), __LINE__, src_name);
    cuda(cudaDeviceReset(), __LINE__, src_name);
}
