#include "framework.h"

const char src_name[] = "channel_pooling.cu";

#define BSUM_BDIM 32

__global__ void batchAverage_K(float *x, float *y, uint length)
{
    __shared__ float temp[BSUM_BDIM];
    uint t_x = threadIdx.x;
    uint n_id = blockIdx.x;
    uint n_offset = length * n_id;

    temp[t_x] = 0;
    for (uint offset = 0; offset < length; offset += BSUM_BDIM) {
        uint index = offset + t_x;
        if (index < length) temp[t_x] += x[n_offset + index];
    }

    uint s = BSUM_BDIM;
    while (s > 1) {
        __syncthreads();
        if (t_x * 2 < s - 1) temp[t_x] += temp[t_x + s / 2];
        if (t_x * 2 == s - 1) temp[t_x] = temp[t_x + s / 2];
        s -= s / 2;
    }

    if (t_x == 0) y[n_id] = temp[0] / length;
}

void forwardChannelPooling(cudaStream_t stream, Node *node, Node *x)
{
    if (x->N != node->N) exitWithInfo(__LINE__, src_name);
    if (x->C != node->C) exitWithInfo(__LINE__, src_name);
    if (node->H != 1) exitWithInfo(__LINE__, src_name);
    if (node->W != 1) exitWithInfo(__LINE__, src_name);
    uint n = x->N * x->C;
    uint length = x->H * x->W;
    dim3 block(BSUM_BDIM);
    dim3 grid(n);
    batchAverage_K<<<grid, block, 0, stream>>>(x->y, node->y, length);
}

__global__ void backwardBatchAverage_K(float *dx, float *dy, uint length)
{
    __shared__ float dy_i;
    uint t_x = threadIdx.x;
    uint n_id = blockIdx.y;
    if (t_x == 0) dy_i = dy[n_id] / length;
    __syncthreads();
    uint n_offset = length * n_id;

    for (uint offset = 0; offset < length; offset += BSUM_BDIM) {
        uint index = offset + t_x;
        if (index < length) dx[n_offset + index] += dy_i;
    }
}

void backwardChannelPooling(cudaStream_t stream, Node *node, Node *x)
{
    if (x->N != node->N) exitWithInfo(__LINE__, src_name);
    if (x->C != node->C) exitWithInfo(__LINE__, src_name);
    if (node->H != 1) exitWithInfo(__LINE__, src_name);
    if (node->W != 1) exitWithInfo(__LINE__, src_name);
    uint n = x->N * x->C;
    uint length = x->H * x->W;
    dim3 block(BSUM_BDIM);
    dim3 grid(n);
    backwardBatchAverage_K<<<grid, block, 0, stream>>>(x->dy, node->dy, length);
}
