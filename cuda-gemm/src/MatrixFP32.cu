#include <assert.h>
#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

MatrixFP32::MatrixFP32(int rows_, int cols_, bool on_device_)
    : rows(rows_), cols(cols_), on_device(on_device_) // constructed for const vars
{
    if (on_device_ == false)
    {
        ptr = new float[rows * cols];
    }
    else
    {
        cudaError_t err = cudaMalloc((void**)&ptr, rows * cols * sizeof(float));
        cuda_check(err);
    }
}

void MatrixFP32::free_mat()
{
    if (on_device == false)
    {
        delete[] ptr;
    }
    else
    {
        cudaFree(ptr);
    }
}

void MatrixFP32::copy_to_device(MatrixFP32 d_mat)
{
    assert(on_device == false && "Matrix must be on host now");
    assert(d_mat.on_device == true && "Input Matrix to this function must be in device memory");

    cudaError_t err = cudaMemcpy(d_mat.ptr, ptr, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cuda_check(err);
}

void MatrixFP32::copy_to_host(MatrixFP32 h_mat)
{
    assert(on_device == true && "Matrix must be on device now");
    assert(h_mat.on_device == false && "Input Matrix to this function must be on host memory");

    cudaError_t err = cudaMemcpy(h_mat.ptr, ptr, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);
}
