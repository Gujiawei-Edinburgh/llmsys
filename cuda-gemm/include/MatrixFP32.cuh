#ifndef MATRIXFP32
#define MATRIXFP32

class MatrixFP32 {
public:
    const int rows;
    const int cols;
    float* ptr;
    const bool on_device;

    MatrixFP32(int rows, int cols, bool on_device);

    void free_mat();

    void copy_to_device(MatrixFP32 d_mat);
    void copy_to_host(MatrixFP32 h_mat);
};

#endif
