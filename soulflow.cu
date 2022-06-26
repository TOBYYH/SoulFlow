#include "framework.h"

const char src_name[] = "soulflow.cu";

typedef struct _sfEncoder {
    // 3 * 192 * 256
    Node conv1; // 16 * 192 * 256 (5x5)
    Node iplu1; // 16 * 192 * 256
    Node pool1; // 16 * 96 * 128
    Node conv2; // 32 * 96 * 128 (5x5)
    Node iplu2; // 32 * 96 * 128
    Node pool2; // 32 * 48 * 64
    Node conv3; // 64 * 48 * 64 (5x5)
    Node iplu3; // 64 * 48 * 64
    Node pool3; // 64 * 24 * 32
    Node conv4; // 128 * 24 * 32 (3x3)
    Node iplu4; // 128 * 24 * 32
    Node pool4; // 128 * 12 * 16
    Node conv5; // 256 * 12 * 16 (3x3)
    Node iplu5; // 256 * 12 * 16
    Node pool5; // 256 * 6 * 8
    Node conv6; // 512 * 6 * 8 (3x3)
    Node iplu6; // 512 * 6 * 8
    Node pool6; // 512 * 3 * 4
} sfEncoder;

void initEncoder(sfEncoder *encoder, Context *context, uint N)
{
    printf("initEncoder: t\n");
    initConvolution2d(context, &(encoder->conv1), 16, 3, 5, 5, N, 192, 256, 1, 2);
    initNode(&(encoder->iplu1), N, 16, 192, 256);
    initMaxpooling2d2x2s2p0(&(encoder->pool1), N, 16, 96, 128);
    initConvolution2d(context, &(encoder->conv2), 32, 16, 5, 5, N, 96, 128, 1, 2);
    initNode(&(encoder->iplu2), N, 32, 96, 128);
    initMaxpooling2d2x2s2p0(&(encoder->pool2), N, 32, 48, 64);
    initConvolution2d(context, &(encoder->conv3), 64, 32, 5, 5, N, 48, 64, 1, 2);
    initNode(&(encoder->iplu3), N, 64, 48, 64);
    initMaxpooling2d2x2s2p0(&(encoder->pool3), N, 64, 24, 32);
    initConvolution2d(context, &(encoder->conv4), 128, 64, 3, 3, N, 24, 32, 1, 1);
    initNode(&(encoder->iplu4), N, 128, 24, 32);
    initMaxpooling2d2x2s2p0(&(encoder->pool4), N, 128, 12, 16);
    initConvolution2d(context, &(encoder->conv5), 256, 128, 3, 3, N, 12, 16, 1, 1);
    initNode(&(encoder->iplu5), N, 256, 12, 16);
    initMaxpooling2d2x2s2p0(&(encoder->pool5), N, 256, 6, 8);
    initConvolution2d(context, &(encoder->conv6), 512, 256, 3, 3, N, 6, 8, 1, 1);
    initNode(&(encoder->iplu6), N, 512, 6, 8);
    initMaxpooling2d2x2s2p0(&(encoder->pool6), N, 512, 3, 4);
}

void freeEncoder(sfEncoder *encoder)
{
    printf("freeEncoder: t\n");
    freeConvolution2d(&(encoder->conv1));
    freeNode(&(encoder->iplu1));
    freePooling(&(encoder->pool1));
    freeConvolution2d(&(encoder->conv2));
    freeNode(&(encoder->iplu2));
    freePooling(&(encoder->pool2));
    freeConvolution2d(&(encoder->conv3));
    freeNode(&(encoder->iplu3));
    freePooling(&(encoder->pool3));
    freeConvolution2d(&(encoder->conv4));
    freeNode(&(encoder->iplu4));
    freePooling(&(encoder->pool4));
    freeConvolution2d(&(encoder->conv5));
    freeNode(&(encoder->iplu5));
    freePooling(&(encoder->pool5));
    freeConvolution2d(&(encoder->conv6));
    freeNode(&(encoder->iplu6));
    freePooling(&(encoder->pool6));
}

typedef struct _sfGRUh {
    Node h; // 1024
    Node fc1; // 1024
    Node iplu; // 1024
    Node fc2; // 1024
} sfGRUh;

void initGRUh(sfGRUh *gru, Context *context, uint N)
{
    gru->h.N = N;
    gru->h.C = 1;
    gru->h.H = 1;
    gru->h.W = 1024;
    cuda(cudaMalloc(&(gru->h.dy), sizeof(float) * 1024), __LINE__, src_name);
    initFullyConnected(context, &(gru->fc1), N, 1024, 1024);
    initNode(&(gru->iplu), N, 1, 1, 1024);
    initFullyConnected(context, &(gru->fc2), N, 1024, 1024);
}

void freeGRUh(sfGRUh *gru)
{
    cuda(cudaFree(gru->h.dy), __LINE__, src_name);
    freeFullyConnected(&(gru->fc1));
    freeNode(&(gru->iplu));
    freeFullyConnected(&(gru->fc2));
}

typedef struct _sfGRUx {
    Node x; // 512 * 3 * 4
    Node fc1; // 1024
    Node iplu; // 1024
    Node fc2; // 1024
} sfGRUx;

void initGRUx(sfGRUx *gru, Context *context, uint N)
{
    gru->x.N = N;
    gru->x.C = 512;
    gru->x.H = 3;
    gru->x.W = 4;
    cuda(cudaMalloc(&(gru->x.dy), sizeof(float) * 512 * 3 * 4), __LINE__, src_name);
    initFullyConnected(context, &(gru->fc1), N, 512*3*4, 1024);
    initNode(&(gru->iplu), N, 1, 1, 1024);
    initFullyConnected(context, &(gru->fc2), N, 1024, 1024);
}

void freeGRUx(sfGRUx *gru)
{
    cuda(cudaFree(gru->x.dy), __LINE__, src_name);
    freeFullyConnected(&(gru->fc1));
    freeNode(&(gru->iplu));
    freeFullyConnected(&(gru->fc2));
}

typedef struct _SoulFlow {
    Context context;
    cudaStream_t *streams;
    cudaEvent_t event;
    uint batch_size;

    float *images_host;
    Node images; // 3 * 192 * 256
    sfEncoder encoder; // 512 * 3 * 4
    Node img_fc1; // 1024
    Node img_iplu; // 1024
    Node img_fc2; // 7

    sfGRUh gru_hr;
    sfGRUx gru_xr;
    Node sigmoid_r;
    sfGRUh gru_hz;
    sfGRUx gru_xz;
    Node sigmoid_z;
    Node h2;
    sfGRUh gru_hh2;
    sfGRUx gru_xh2;
    Node tanh_h2;

    float **x;
    float **h;
    uint h_id;
    uint h_size;

    Node ht;
    Node out_fc1; // 1024
    Node out_iplu; // 1024
    Node out_fc2; // 7
} SoulFlow;

#define STREAM_NUM 3

extern "C"
SoulFlow *initSoulFlow(uint N)
{
    printf("initSoulFlow: t\n");
    SoulFlow *sf = (SoulFlow *)malloc(sizeof(SoulFlow));

    initContext(&(sf->context));
    sf->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * STREAM_NUM);
    for (uint i = 0; i < STREAM_NUM; i++)
        cuda(cudaStreamCreate(&(sf->streams[i])), __LINE__, src_name);
    cuda(cudaEventCreate(&(sf->event)), __LINE__, src_name);
    sf->batch_size = N;

    sf->images_host = (float *)malloc(sizeof(float) * N * 3 * 192 * 256);
    initNode(&(sf->images), N, 3, 192, 256);
    initEncoder(&(sf->encoder), &(sf->context), N);
    initFullyConnected(&(sf->context), &(sf->img_fc1), N, 512*3*4, 1024);
    initNode(&(sf->img_iplu), N, 1, 1, 1024);
    initFullyConnected(&(sf->context), &(sf->img_fc2), N, 1024, 7);

    initGRUh(&(sf->gru_hr), &(sf->context), N);
    initGRUx(&(sf->gru_xr), &(sf->context), N);
    initNode(&(sf->sigmoid_r), N, 1, 1, 1024);
    initGRUh(&(sf->gru_hz), &(sf->context), N);
    initGRUx(&(sf->gru_xz), &(sf->context), N);
    initNode(&(sf->sigmoid_z), N, 1, 1, 1024);
    initNode(&(sf->h2), N, 1, 1, 1024);
    initGRUh(&(sf->gru_hh2), &(sf->context), N);
    sf->gru_hh2.h.y = sf->h2.y;
    initGRUx(&(sf->gru_xh2), &(sf->context), N);
    initNode(&(sf->tanh_h2), N, 1, 1, 1024);

    sf->x = (float **)malloc(sizeof(void *) * 10);
    sf->h = (float **)malloc(sizeof(void *) * 10);
    for (uint i = 0; i < 10; i++) {
        sf->x[i] = NULL;
        sf->h[i] = NULL;
    }
    size_t size = sizeof(float) * N * 1024;
    cuda(cudaMalloc(&(sf->h[0]), size), __LINE__, src_name);
    cuda(cudaMemset(sf->h[0], 0, size), __LINE__, src_name);
    sf->h_id = 0;
    sf->h_size = 10;

    sf->ht.N = N;
    sf->ht.C = 1;
    sf->ht.H = 1;
    sf->ht.W = 1024;
    cuda(cudaMalloc(&(sf->ht.dy), sizeof(float) * 1024), __LINE__, src_name);
    initFullyConnected(&(sf->context), &(sf->out_fc1), N, 1024, 1024);
    initNode(&(sf->out_iplu), N, 1, 1, 1024);
    initFullyConnected(&(sf->context), &(sf->out_fc2), N, 1024, 7);

    printf("Work space size: %ld MB\n", sf->context.space_size/(1024*1024));
    cuda(cudaMalloc(&(sf->context.work_space), sf->context.space_size), __LINE__, src_name);

    return sf;
}

extern "C"
void freeSoulFlow(SoulFlow *sf)
{
    printf("freeSoulFlow: t\n");
    freeContext(&(sf->context));
    cuda(cudaFree(sf->context.work_space), __LINE__, src_name);
    for (uint i = 0; i < STREAM_NUM; i++)
        cuda(cudaStreamDestroy(sf->streams[i]), __LINE__, src_name);
    free(sf->streams);
    cuda(cudaEventDestroy(sf->event), __LINE__, src_name);

    free(sf->images_host);
    freeNode(&(sf->images));
    freeEncoder(&(sf->encoder));
    freeFullyConnected(&(sf->img_fc1));
    freeNode(&(sf->img_iplu));
    freeFullyConnected(&(sf->img_fc2));

    freeGRUh(&(sf->gru_hr));
    freeGRUx(&(sf->gru_xr));
    freeNode(&(sf->sigmoid_r));
    freeGRUh(&(sf->gru_hz));
    freeGRUx(&(sf->gru_xz));
    freeNode(&(sf->sigmoid_z));
    freeNode(&(sf->h2));
    freeGRUh(&(sf->gru_hh2));
    freeGRUx(&(sf->gru_xh2));
    freeNode(&(sf->tanh_h2));

    for (uint i = 0; i < sf->h_size; i++) {
        if (sf->x[i] != NULL)
            cuda(cudaFree(sf->x[i]), __LINE__, src_name);
        if (sf->h[i] != NULL)
            cuda(cudaFree(sf->h[i]), __LINE__, src_name);
    }
    free(sf->x);
    free(sf->h);

    cuda(cudaFree(sf->ht.dy), __LINE__, src_name);
    freeFullyConnected(&(sf->out_fc1));
    freeNode(&(sf->out_iplu));
    freeFullyConnected(&(sf->out_fc2));

    free(sf);
}

extern "C"
uint sfGetWeightNum(SoulFlow *sf)
{
    return sf->context.W_num;
}

extern "C"
uint sfGetWeightLength(SoulFlow *sf, uint index)
{
    if (index >= sf->context.W_num) {
        printf("sfGetWeightLength: Index over bound.\n");
        return 0;
    }
    return sf->context.W_list[index]->length;
}

extern "C"
void sfCopyWeight(SoulFlow *sf, uint index, float *host, int mode)
{
    if (index >= sf->context.W_num) {
        printf("sfCopyWeight: Index over bound.\n");
        return;
    }
    TensorW *tensor = sf->context.W_list[index];
    float *device = tensor->_tensor_;
    size_t size = sizeof(float) * tensor->length;
    if (mode)
        cuda(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice), __LINE__, src_name);
    else
        cuda(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost), __LINE__, src_name);
}

extern "C"
void sfSetGradientToZero(SoulFlow *sf)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
    setGradientToZero(&(sf->context));
}

extern "C"
void sfSetSample(SoulFlow *sf, uint N_id, uint8_t *image)
{
    if (N_id >= sf->batch_size) exitWithInfo(__LINE__, src_name);

    uint N_offset = 3 * 192 * 256 * N_id;
    for (uint h = 0; h < 192; h++) {
        uint src_offset1 = 3 * 256 * h;
        uint dst_offset1 = N_offset + 256 * h;
        for (uint w = 0; w < 256; w++) {
            uint src_offset2 = src_offset1 + 3 * w;
            uint dst_offset2 = dst_offset1 + w;
            sf->images_host[dst_offset2] = (float)image[src_offset2 + 2] / 255.0f;
            sf->images_host[dst_offset2+192*256] = (float)image[src_offset2 + 1] / 255.0f;
            sf->images_host[dst_offset2+192*256*2] = (float)image[src_offset2] / 255.0f;
        }
    }
}

void forwardEncoder(SoulFlow *sf)
{
    Context *context = &(sf->context);
    sfEncoder *encoder = &(sf->encoder);
    forwardConvolution2d(context, &(encoder->conv1), &(sf->images));
    forwardIPLU(context->stream, &(encoder->iplu1), &(encoder->conv1));
    forwardPooling(context, &(encoder->pool1), &(encoder->iplu1));
    forwardConvolution2d(context, &(encoder->conv2), &(encoder->pool1));
    forwardIPLU(context->stream, &(encoder->iplu2), &(encoder->conv2));
    forwardPooling(context, &(encoder->pool2), &(encoder->iplu2));
    forwardConvolution2d(context, &(encoder->conv3), &(encoder->pool2));
    forwardIPLU(context->stream, &(encoder->iplu3), &(encoder->conv3));
    forwardPooling(context, &(encoder->pool3), &(encoder->iplu3));
    forwardConvolution2d(context, &(encoder->conv4), &(encoder->pool3));
    forwardIPLU(context->stream, &(encoder->iplu4), &(encoder->conv4));
    forwardPooling(context, &(encoder->pool4), &(encoder->iplu4));
    forwardConvolution2d(context, &(encoder->conv5), &(encoder->pool4));
    forwardIPLU(context->stream, &(encoder->iplu5), &(encoder->conv5));
    forwardPooling(context, &(encoder->pool5), &(encoder->iplu5));
    forwardConvolution2d(context, &(encoder->conv6), &(encoder->pool5));
    forwardIPLU(context->stream, &(encoder->iplu6), &(encoder->conv6));
    forwardPooling(context, &(encoder->pool6), &(encoder->iplu6));
}

extern "C"
void forwardImage(SoulFlow *sf, float *output)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);

    size_t size = sizeof(float) * sf->batch_size * 3 * 192 * 256;
    cuda(cudaMemcpy(sf->images.y, sf->images_host, size,
                    cudaMemcpyHostToDevice), __LINE__, src_name);
    
    setStream(&(sf->context), sf->streams[0]);
    forwardEncoder(sf);
    forwardFullyConnected(&(sf->context), &(sf->img_fc1), &(sf->encoder.pool6));
    forwardIPLU(sf->context.stream, &(sf->img_iplu), &(sf->img_fc1));
    forwardFullyConnected(&(sf->context), &(sf->img_fc2), &(sf->img_iplu));

    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
    cuda(cudaMemcpy(output, sf->img_fc2.y, sizeof(float) * sf->batch_size * 7,
                    cudaMemcpyDeviceToHost), __LINE__, src_name);
}

void backwardEncoder(SoulFlow *sf)
{
    Context *context = &(sf->context);
    sfEncoder *encoder = &(sf->encoder);
    backwardPooling(context, &(encoder->pool6), &(encoder->iplu6));
    backwardIPLU(context->stream, &(encoder->iplu6), &(encoder->conv6));
    backwardConvolution2d(context, &(encoder->conv6), &(encoder->pool5));
    backwardPooling(context, &(encoder->pool5), &(encoder->iplu5));
    backwardIPLU(context->stream, &(encoder->iplu5), &(encoder->conv5));
    backwardConvolution2d(context, &(encoder->conv5), &(encoder->pool4));
    backwardPooling(context, &(encoder->pool4), &(encoder->iplu4));
    backwardIPLU(context->stream, &(encoder->iplu4), &(encoder->conv4));
    backwardConvolution2d(context, &(encoder->conv4), &(encoder->pool3));
    backwardPooling(context, &(encoder->pool3), &(encoder->iplu3));
    backwardIPLU(context->stream, &(encoder->iplu3), &(encoder->conv3));
    backwardConvolution2d(context, &(encoder->conv3), &(encoder->pool2));
    backwardPooling(context, &(encoder->pool2), &(encoder->iplu2));
    backwardIPLU(context->stream, &(encoder->iplu2), &(encoder->conv2));
    backwardConvolution2d(context, &(encoder->conv2), &(encoder->pool1));
    backwardPooling(context, &(encoder->pool1), &(encoder->iplu1));
    backwardIPLU(context->stream, &(encoder->iplu1), &(encoder->conv1));
    backwardConvolution2d(context, &(encoder->conv1), &(sf->images));
}

extern "C"
void backwardImage(SoulFlow *sf, float *gradient)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);

    cuda(cudaMemcpy(sf->img_fc2.dy, gradient, sizeof(float) * sf->batch_size * 7,
                    cudaMemcpyHostToDevice), __LINE__, src_name);
    
    setStream(&(sf->context), sf->streams[0]);
    backwardFullyConnected(&(sf->context), &(sf->img_fc2), &(sf->img_iplu));
    backwardIPLU(sf->context.stream, &(sf->img_iplu), &(sf->img_fc1));
    backwardFullyConnected(&(sf->context), &(sf->img_fc1), &(sf->encoder.pool6));
    backwardEncoder(sf);
}

extern "C"
void momentunImage(SoulFlow *sf, float m, float lr)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
    Convolution *conv;
    conv = (Convolution *)sf->encoder.conv1._content_;
    momentunTensorW(&(conv->filter), m, lr);
    momentunTensorW(&(conv->bias), m, lr);
    conv = (Convolution *)sf->encoder.conv2._content_;
    momentunTensorW(&(conv->filter), m, lr);
    momentunTensorW(&(conv->bias), m, lr);
    conv = (Convolution *)sf->encoder.conv3._content_;
    momentunTensorW(&(conv->filter), m, lr);
    momentunTensorW(&(conv->bias), m, lr);
    conv = (Convolution *)sf->encoder.conv4._content_;
    momentunTensorW(&(conv->filter), m, lr);
    momentunTensorW(&(conv->bias), m, lr);
    conv = (Convolution *)sf->encoder.conv5._content_;
    momentunTensorW(&(conv->filter), m, lr);
    momentunTensorW(&(conv->bias), m, lr);
    conv = (Convolution *)sf->encoder.conv6._content_;
    momentunTensorW(&(conv->filter), m, lr);
    momentunTensorW(&(conv->bias), m, lr);
    FullyConnected *fc;
    fc = (FullyConnected *)sf->img_fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->img_fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
}

void forwardGRUh(SoulFlow *sf, uint index)
{
    Context *context = &(sf->context);
    sfGRUh *gru_h;
    switch (index) {
        case 1: gru_h = &(sf->gru_hr); gru_h->h.y = sf->h[sf->h_id]; break;
        case 2: gru_h = &(sf->gru_hz); gru_h->h.y = sf->h[sf->h_id]; break;
        case 3: gru_h = &(sf->gru_hh2);
    }
    forwardFullyConnected(context, &(gru_h->fc1), &(gru_h->h));
    forwardIPLU(context->stream, &(gru_h->iplu), &(gru_h->fc1));
    forwardFullyConnected(&(sf->context), &(gru_h->fc2), &(gru_h->iplu));
}

void forwardGRUx(SoulFlow *sf, uint index)
{
    Context *context = &(sf->context);
    sfGRUx *gru_x;
    switch (index) {
        case 1: gru_x = &(sf->gru_xr); break;
        case 2: gru_x = &(sf->gru_xz); break;
        case 3: gru_x = &(sf->gru_xh2);
    }
    gru_x->x.y = sf->x[sf->h_id];
    forwardFullyConnected(context, &(gru_x->fc1), &(gru_x->x));
    forwardIPLU(context->stream, &(gru_x->iplu), &(gru_x->fc1));
    forwardFullyConnected(&(sf->context), &(gru_x->fc2), &(gru_x->iplu));
}

__global__ void forward_h_K(float *h, float *z, float *tanh_h2, float *ht)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    float z_i = z[id];
    ht[id] = tanh_h2[id] * z_i + h[id] * (1 - z_i);
}

extern "C"
void forwardGRU(SoulFlow *sf, int train)
{
    if (sf->h_id + 1 == sf->h_size) {
        uint new_size = sf->h_size + 10;
        sf->x = (float **)realloc(sf->x, sizeof(void *) * new_size);
        sf->h = (float **)realloc(sf->h, sizeof(void *) * new_size);
        for (uint i = sf->h_size; i < new_size; i++) {
            sf->x[i] = NULL;
            sf->h[i] = NULL;
        }
        sf->h_size = new_size;
    }
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);

    size_t size = sizeof(float) * sf->batch_size * 512 * 3 * 4;
    if (sf->x[sf->h_id] == NULL)
        cuda(cudaMalloc(&(sf->x[sf->h_id]), size), __LINE__, src_name);
    size = sizeof(float) * sf->batch_size * 1024;
    if (sf->h[sf->h_id + 1] == NULL)
        cuda(cudaMalloc(&(sf->h[sf->h_id + 1]), size), __LINE__, src_name);

    size = sizeof(float) * sf->batch_size * 3 * 192 * 256;
    cuda(cudaMemcpy(sf->images.y, sf->images_host, size,
                    cudaMemcpyHostToDevice), __LINE__, src_name);
    
    setStream(&(sf->context), sf->streams[0]);
    forwardEncoder(sf);
    tensorOptionBlocking(4, sf->x[sf->h_id], sf->encoder.pool6.y, NULL, sf->batch_size*512*3*4);
    forwardGRUh(sf, 1);
    forwardGRUx(sf, 1);
    tensorOption(sf->streams[0], 2, sf->gru_hr.fc2.y,
                 sf->gru_hr.fc2.y, sf->gru_xr.fc2.y, sf->batch_size*1024);
    forwardSigmoid(sf->streams[0], &(sf->sigmoid_r), &(sf->gru_hr.fc2));
    tensorOption(sf->streams[0], 1, sf->h2.y,
                 sf->h[sf->h_id], sf->sigmoid_r.y, sf->batch_size*1024);
    forwardGRUx(sf, 3);

    setStream(&(sf->context), sf->streams[1]);
    forwardGRUh(sf, 2);
    forwardGRUx(sf, 2);
    tensorOption(sf->streams[1], 2, sf->gru_hz.fc2.y,
                 sf->gru_hz.fc2.y, sf->gru_xz.fc2.y, sf->batch_size*1024);
    forwardSigmoid(sf->streams[1], &(sf->sigmoid_z), &(sf->gru_hz.fc2));

    setStream(&(sf->context), sf->streams[2]);
    forwardGRUx(sf, 3);
    tensorOptionBlocking(2, sf->gru_hh2.fc2.y, sf->gru_hh2.fc2.y,
                         sf->gru_xh2.fc2.y, sf->batch_size*1024);
    forwardTanh(sf->streams[2], &(sf->tanh_h2), &(sf->gru_hh2.fc2));
    forward_h_K<<<16, 32>>>(sf->h[sf->h_id], sf->sigmoid_z.y, sf->tanh_h2.y, sf->h[sf->h_id + 1]);

    if (train) sf->h_id += 1;
    else {
        float *ht = sf->h[sf->h_id];
        sf->h[sf->h_id] = sf->h[sf->h_id + 1];
        sf->h[sf->h_id + 1] = ht;
    }
}

extern "C"
void predictGRU(SoulFlow *sf, float *output)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);

    setStream(&(sf->context), sf->streams[0]);
    sf->ht.y = sf->h[sf->h_id];
    forwardFullyConnected(&(sf->context), &(sf->out_fc1), &(sf->ht));
    forwardIPLU(sf->streams[0], &(sf->out_iplu), &(sf->out_fc1));
    forwardFullyConnected(&(sf->context), &(sf->out_fc2), &(sf->out_iplu));

    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
    cuda(cudaMemcpy(output, sf->out_fc2.y, sizeof(float) * sf->batch_size * 7,
                    cudaMemcpyDeviceToHost), __LINE__, src_name);
    
    printf("NN output:\n");
    for (uint n = 0; n < sf->batch_size; n++) {
        for (uint i = 0; i < 7; i++) printf("%g ", output[7 * n + i]);
        printf("\n");
    }
}

void backwardGRUh(SoulFlow *sf, uint index)
{
    Context *context = &(sf->context);
    sfGRUh *gru_h;
    switch (index) {
        case 1: gru_h = &(sf->gru_hr); gru_h->h.y = sf->h[sf->h_id]; break;
        case 2: gru_h = &(sf->gru_hz); gru_h->h.y = sf->h[sf->h_id]; break;
        case 3: gru_h = &(sf->gru_hh2);
    }
    backwardFullyConnected(&(sf->context), &(gru_h->fc2), &(gru_h->iplu));
    backwardIPLU(context->stream, &(gru_h->iplu), &(gru_h->fc1));
    backwardFullyConnected(context, &(gru_h->fc1), &(gru_h->h));
}

void backwardGRUx(SoulFlow *sf, uint index)
{
    Context *context = &(sf->context);
    sfGRUx *gru_x;
    switch (index) {
        case 1: gru_x = &(sf->gru_xr); break;
        case 2: gru_x = &(sf->gru_xz); break;
        case 3: gru_x = &(sf->gru_xh2);
    }
    gru_x->x.y = sf->x[sf->h_id];
    backwardFullyConnected(&(sf->context), &(gru_x->fc2), &(gru_x->iplu));
    backwardIPLU(context->stream, &(gru_x->iplu), &(gru_x->fc1));
    backwardFullyConnected(context, &(gru_x->fc1), &(gru_x->x));
}

__global__ void backward_h2_K(float *h, float *z, float *tanh_h2,
                              float *dz, float *dtanh_h2, float *dh)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    float dh_i = dh[id];
    float z_i = z[id];
    dz[id] = (tanh_h2[id] - h[id]) * dh_i;
    dtanh_h2[id] = z_i * dh_i;
    dh[id] = (1 - z_i) * dh_i;
}

__global__ void backward_h_K(float *dhr, float *dhz, float *dhh2, float *dh, float *r)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    float dh_i = dhh2[id] * r[id] + dhr[id] + dhz[id];
    dh[id] = dh[id] + dh_i;
}

extern "C"
void backwardGRU(SoulFlow *sf, float *gradient)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
    cuda(cudaMemcpy(sf->out_fc2.dy, gradient, sizeof(float) * sf->batch_size * 7,
                    cudaMemcpyHostToDevice), __LINE__, src_name);
    
    setStream(&(sf->context), sf->streams[0]);
    backwardFullyConnected(&(sf->context), &(sf->out_fc2), &(sf->out_iplu));
    backwardIPLU(sf->streams[0], &(sf->out_iplu), &(sf->out_fc1));
    backwardFullyConnected(&(sf->context), &(sf->out_fc1), &(sf->ht));

    sf->h_id -= 1;

    while (1) {
        // printf("backwardGRU: %d\n", sf->h_id);
        backward_h2_K<<<16, 32>>>(sf->h[sf->h_id], sf->sigmoid_z.y, sf->tanh_h2.y,
                                  sf->sigmoid_z.dy, sf->tanh_h2.dy, sf->ht.dy);
        // cuda(cudaDeviceSynchronize(), __LINE__, src_name);/////////////////////////////////////////
        
        setStream(&(sf->context), sf->streams[0]);
        backwardSigmoid(sf->streams[0], &(sf->sigmoid_z), &(sf->gru_hz.fc2));
        tensorOption(sf->streams[0], 4, sf->gru_xz.fc2.dy,
                     sf->gru_hz.fc2.dy, NULL, sf->batch_size*1024);
        backwardGRUh(sf, 2);
        backwardGRUx(sf, 2);
        // cuda(cudaDeviceSynchronize(), __LINE__, src_name);/////////////////////////////////////////

        setStream(&(sf->context), sf->streams[1]);
        backwardTanh(sf->streams[1], &(sf->tanh_h2), &(sf->gru_hh2.fc2));
        tensorOption(sf->streams[1], 4, sf->gru_xh2.fc2.dy,
                     sf->gru_hh2.fc2.dy, NULL, sf->batch_size*1024);
        backwardGRUh(sf, 3);
        backwardGRUx(sf, 3);
        tensorOption(sf->streams[1], 1, sf->sigmoid_r.dy, sf->h[sf->h_id],
                     sf->gru_hh2.h.dy, sf->batch_size*1024);
        backwardSigmoid(sf->streams[1], &(sf->sigmoid_r), &(sf->gru_hr.fc2));
        tensorOption(sf->streams[1], 4, sf->gru_xr.fc2.dy,
                     sf->gru_hr.fc2.dy, NULL, sf->batch_size*1024);
        backwardGRUh(sf, 1);
        backwardGRUx(sf, 1);
        backward_h_K<<<16, 32>>>(sf->gru_hr.h.dy, sf->gru_hz.h.dy,
                                 sf->gru_hh2.h.dy, sf->ht.dy, sf->sigmoid_r.y);
        cuda(cudaDeviceSynchronize(), __LINE__, src_name);/////////////////////////////////////////

        if (sf->h_id == 1) break;

        sf->h_id -= 1;
        setStream(&(sf->context), sf->streams[0]);
        forwardGRUh(sf, 1);
        forwardGRUx(sf, 1);
        tensorOption(sf->streams[0], 2, sf->gru_hr.fc2.y,
                     sf->gru_hr.fc2.y, sf->gru_xr.fc2.y, sf->batch_size*1024);
        forwardSigmoid(sf->streams[0], &(sf->sigmoid_r), &(sf->gru_hr.fc2));
        tensorOption(sf->streams[0], 1, sf->h2.y,
                     sf->h[sf->h_id], sf->sigmoid_r.y, sf->batch_size*1024);
        forwardGRUx(sf, 3);

        setStream(&(sf->context), sf->streams[1]);
        forwardGRUh(sf, 2);
        forwardGRUx(sf, 2);
        tensorOption(sf->streams[1], 2, sf->gru_hz.fc2.y,
                     sf->gru_hz.fc2.y, sf->gru_xz.fc2.y, sf->batch_size*1024);
        forwardSigmoid(sf->streams[1], &(sf->sigmoid_z), &(sf->gru_hz.fc2));

        setStream(&(sf->context), sf->streams[2]);
        forwardGRUx(sf, 3);
        tensorOptionBlocking(2, sf->gru_hh2.fc2.y, sf->gru_hh2.fc2.y,
                             sf->gru_xh2.fc2.y, sf->batch_size*1024);
        forwardTanh(sf->streams[2], &(sf->tanh_h2), &(sf->gru_hh2.fc2));
    }
}

extern "C"
void momentunGRU(SoulFlow *sf, float m, float lr)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
    FullyConnected *fc;
    fc = (FullyConnected *)sf->gru_hr.fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_hr.fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_xr.fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_xr.fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_hz.fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_hz.fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_xz.fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_xz.fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_hh2.fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_hh2.fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_xh2.fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->gru_xh2.fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->out_fc1._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
    fc = (FullyConnected *)sf->out_fc2._content_;
    momentunTensorW(&(fc->W), m, lr);
    momentunTensorW(&(fc->b), m, lr);
}

extern "C"
void resetTemp(SoulFlow *sf)
{
    sf->h_id = 0;
}

extern "C"
void deviceSynchronize(void)
{
    cuda(cudaDeviceSynchronize(), __LINE__, src_name);
}

extern "C"
uint sfHookTensor(SoulFlow *sf, float *ndarray)
{
    // Convolution *conv = (Convolution *)sf->attention.conv2._content_;
    // float *d_ptr = conv->filter._tensor_;
    // uint length = conv->filter.length;

    // Node *node = &(sf->encoder.conv5);
    // float *d_ptr = node->y;
    // uint length = node->N * node->C * node->H * node->W;

    FullyConnected *fc = (FullyConnected *)sf->gru_hr.fc1._content_;
    float *d_ptr = fc->W._tensor_;
    uint length = fc->W.length;

    if (ndarray == NULL) return length;
    size_t size = sizeof(float) * length;
    cuda(cudaMemcpy(ndarray, d_ptr, size, cudaMemcpyDeviceToHost), __LINE__, src_name);
    return 0;
}
