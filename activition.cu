#include "framework.h"

const char src_name[] = "activition.cu";

__global__ void forwardTanh_K(float *x, float *y, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    float ex = exp(x[id]);
    y[id] = 1 - 2 / (ex * ex + 1);
}

void forwardTanh(cudaStream_t stream, Node *node, Node *x)
{
    if (node->N != x->N) exitWithInfo(__LINE__, src_name);
    if (node->C != x->C) exitWithInfo(__LINE__, src_name);
    if (node->H != x->H) exitWithInfo(__LINE__, src_name);
    if (node->W != x->W) exitWithInfo(__LINE__, src_name);
    uint length = node->N * node->C * node->H * node->W;
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    forwardTanh_K<<<grid, block, 0, stream>>>(x->y, node->y, length);
}

__global__ void backwardTanh_K(float *dx, float *dy, float *y, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    float t = y[id];
    dx[id] = (1 - t * t) * dy[id];
}

void backwardTanh(cudaStream_t stream, Node *node, Node *x)
{
    if (node->N != x->N) exitWithInfo(__LINE__, src_name);
    if (node->C != x->C) exitWithInfo(__LINE__, src_name);
    if (node->H != x->H) exitWithInfo(__LINE__, src_name);
    if (node->W != x->W) exitWithInfo(__LINE__, src_name);
    uint length = node->N * node->C * node->H * node->W;
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    backwardTanh_K<<<grid, block, 0, stream>>>(x->dy, node->dy, node->y, length);
}

// ================================================================================================
__global__ void forwardSigmoid_K(float *x, float *y, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    float ex = exp(x[id]);
    y[id] = ex / (ex + 1);
}

void forwardSigmoid(cudaStream_t stream, Node *node, Node *x)
{
    if (node->N != x->N) exitWithInfo(__LINE__, src_name);
    if (node->C != x->C) exitWithInfo(__LINE__, src_name);
    if (node->H != x->H) exitWithInfo(__LINE__, src_name);
    if (node->W != x->W) exitWithInfo(__LINE__, src_name);
    uint length = node->N * node->C * node->H * node->W;
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    forwardSigmoid_K<<<grid, block, 0, stream>>>(x->y, node->y, length);
}

__global__ void backwardSigmoid_K(float *dx, float *dy, float *y, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    float t = y[id];
    dx[id] = t * (1 - t) * dy[id];
}

void backwardSigmoid(cudaStream_t stream, Node *node, Node *x)
{
    if (node->N != x->N) exitWithInfo(__LINE__, src_name);
    if (node->C != x->C) exitWithInfo(__LINE__, src_name);
    if (node->H != x->H) exitWithInfo(__LINE__, src_name);
    if (node->W != x->W) exitWithInfo(__LINE__, src_name);
    uint length = node->N * node->C * node->H * node->W;
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    backwardSigmoid_K<<<grid, block, 0, stream>>>(x->dy, node->dy, node->y, length);
}

// ================================================================================================
__global__ void forwardIPLU_K(float *x, float *y, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    float t = x[id];
    if (t >=0) y[id] = t;
    else y[id] = t / (1.0f - t);
}

void forwardIPLU(cudaStream_t stream, Node *node, Node *x)
{
    if (node->N != x->N) exitWithInfo(__LINE__, src_name);
    if (node->C != x->C) exitWithInfo(__LINE__, src_name);
    if (node->H != x->H) exitWithInfo(__LINE__, src_name);
    if (node->W != x->W) exitWithInfo(__LINE__, src_name);
    uint length = node->N * node->C * node->H * node->W;
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    forwardIPLU_K<<<grid, block, 0, stream>>>(x->y, node->y, length);
}

__global__ void backwardIPLU_K(float *dx, float *dy, float *x, uint length)
{
    uint id = BLOCKDIM_X * blockIdx.x + threadIdx.x;
    if (id >= length) return;
    float t = x[id];
    if (t >= 0) dx[id] = dy[id];
    else {
        float tt = 1.0f - t;
        dx[id] = dy[id] / (tt * tt);
    }
}

void backwardIPLU(cudaStream_t stream, Node *node, Node *x)
{
    if (node->N != x->N) exitWithInfo(__LINE__, src_name);
    if (node->C != x->C) exitWithInfo(__LINE__, src_name);
    if (node->H != x->H) exitWithInfo(__LINE__, src_name);
    if (node->W != x->W) exitWithInfo(__LINE__, src_name);
    uint length = node->N * node->C * node->H * node->W;
    uint gridx = length / BLOCKDIM_X;
    if (length % BLOCKDIM_X != 0) gridx += 1;
    dim3 block(BLOCKDIM_X);
    dim3 grid(gridx);
    backwardIPLU_K<<<grid, block, 0, stream>>>(x->dy, node->dy, x->y, length);
}
