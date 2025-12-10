#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// ------------------------------------------------------------
// Global datatype for CNN computation
// ------------------------------------------------------------
typedef float TYPE;   // Fashion-MNIST uses float32 weights and inputs

// ------------------------------------------------------------
// Convolution Kernel Argument Structure
//
// Matches your conv kernel:
//   - C_in = 1 for grayscale Fashion-MNIST
//   - C_out = 8 for your trained model
//   - padding = 0 (Keras "valid")
//   - stride = 1
//   - grid_dim = {W_out, H_out, 1}
// ------------------------------------------------------------
typedef struct {
    uint64_t I_addr;      // device address of input feature map
    uint64_t W_addr;      // device address of conv weights
    uint64_t B_addr;      // device address of biases
    uint64_t O_addr;      // device address of output feature map

    int32_t C_in;         // input channels (1)
    int32_t C_out;        // output channels (8)
    int32_t H;            // input height
    int32_t W;            // input width

    int32_t K;            // filter size (3)
    int32_t padding;      
    int32_t stride;

    int32_t H_out;
    int32_t W_out;

    int32_t grid_dim[3];  // x, y, z
    int32_t block_dim[3]; // x, y, z
} kernel_arg_conv_t;

// ------------------------------------------------------------
// MaxPool Kernel Argument Structure
//
// pool_size = 2
// output dims = (C, H/2, W/2)
// ------------------------------------------------------------
typedef struct {
    uint64_t I_addr;      // input feature map
    uint64_t O_addr;      // output feature map

    int32_t C;
    int32_t H;
    int32_t W;

    int32_t grid_dim[3];
    int32_t block_dim[3];
} kernel_arg_pool_t;

// ------------------------------------------------------------
// ReLU Kernel Argument Structure
//
// Applies ReLU to entire tensor elementwise
// ------------------------------------------------------------
typedef struct {
    uint64_t I_addr;  
    uint64_t O_addr;  

    int32_t C;
    int32_t H;
    int32_t W;

    int32_t grid_dim[3]; 
    int32_t block_dim[3];
} kernel_arg_relu_t;

// ------------------------------------------------------------
// HOST-SIDE Tensor struct (not used on device)
// ------------------------------------------------------------
struct Tensor {
    int C, H, W;
    std::vector<float> data;

    Tensor(int c=0, int h=0, int w=0)
        : C(c), H(h), W(w), data(c*h*w) {}

    inline float& at(int c, int y, int x) {
        return data[c*H*W + y*W + x];
    }
};

#endif // COMMON_H
