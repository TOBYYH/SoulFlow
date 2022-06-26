cuda_includes = /usr/local/cuda-11.6/include
cuda_librarys = /usr/local/cuda-11.6/lib64
VPATH = obj

heads = framework.h
srcs = framework.cu activition.cu channel_pooling.cu upsample2x.cu soulflow.cu
objs = $(patsubst %.cu, %.o, $(srcs))
libs = -lcudart -lcublas -lcudnn

libsoulflow.so: $(objs)
	gcc obj/*.o -shared -o libsoulflow.so $(libs) -L$(cuda_librarys)

$(objs): %.o: %.cu $(heads)
	nvcc -arch=sm_80 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 \
	-gencode=arch=compute_86,code=compute_86 \
	-c -O3 -Xptxas -dlcm=cg -Xcompiler -fPIC $< -o obj/$@ -I$(cuda_includes)
