// main_cnn.cpp
#include <iostream>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <vortex.h>

#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr) do {                       \
    int _ret = _expr;                              \
    if (_ret) {                                    \
        printf("Error: %s returned %d\n",          \
               #_expr, _ret);                      \
        cleanup();                                 \
        exit(-1);                                  \
    }                                              \
} while (false)

// ------------------------------------------------------------
// Layer sizes — simplest real CNN possible
// ------------------------------------------------------------
static const int C_IN  = 3;
static const int C_OUT = 4;   // Conv layer output channels
static const int K     = 3;

static const int INPUT_SIZE = 32;   // H = W = 32

// ------------------------------------------------------------
// CPU reference functionality (conv → relu → pool)
// ------------------------------------------------------------

static void conv_cpu(
    std::vector<float>& O,
    const std::vector<float>& I,
    const std::vector<float>& W,
    int C_in, int C_out, int H, int Wt,
    int padding, int stride)
{
    const int K = 3;
    int H_out = (H + 2*padding - K) / stride + 1; 
    int W_out = (Wt + 2*padding - K) / stride + 1;

    for (int oc = 0; oc < C_out; oc++) {
        for (int oy = 0; oy < H_out; oy++) {
            for (int ox = 0; ox < W_out; ox++) {

                float sum = 0.0f;

                for (int ic = 0; ic < C_in; ic++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {

                            int iy = oy - padding + ky;
                            int ix = ox - padding + kx;

                            if (iy >= 0 && iy < H && ix >= 0 && ix < Wt) {
                                int in_idx = ic*H*Wt + iy*Wt + ix;
                                int wt_idx = ((oc*C_in + ic)*K + ky)*K + kx;
                                sum += I[in_idx] * W[wt_idx];
                            }
                        }
                    }
                }

                O[oc*H_out*W_out + oy*W_out + ox] = sum;
            }
        }
    }
}

static void relu_cpu(std::vector<float>& X) {
    for (auto& v : X)
        if (v < 0) v = 0;
}

static void maxpool_cpu(
    std::vector<float>& O,
    const std::vector<float>& I,
    int C, int H, int W)
{
    int H_out = H / 2;
    int W_out = W / 2;

    for (int oc = 0; oc < C; oc++) {
        for (int oy = 0; oy < H_out; oy++) {
            for (int ox = 0; ox < W_out; ox++) {

                float m = -1e30f;

                for (int ky = 0; ky < 2; ky++) {
                    for (int kx = 0; kx < 2; kx++) {
                        int iy = 2*oy + ky;
                        int ix = 2*ox + kx;
                        float v = I[oc*H*W + iy*W + ix];
                        if (v > m) m = v;
                    }
                }

                O[oc*H_out*W_out + oy*W_out + ox] = m;
            }
        }
    }
}


// ------------------------------------------------------------
// Device globals
// ------------------------------------------------------------
vx_device_h device = nullptr;

vx_buffer_h I_buf = nullptr;
vx_buffer_h W_buf = nullptr;
vx_buffer_h O1_buf = nullptr;  // conv output
vx_buffer_h O2_buf = nullptr;  // pool output

vx_buffer_h conv_bin = nullptr;
vx_buffer_h relu_bin = nullptr;
vx_buffer_h pool_bin = nullptr;

vx_buffer_h args_buf = nullptr;

// cleanup helper
void cleanup() {
    if (device) {
        vx_mem_free(I_buf);
        vx_mem_free(W_buf);
        vx_mem_free(O1_buf);
        vx_mem_free(O2_buf);
        vx_mem_free(conv_bin);
        vx_mem_free(relu_bin);
        vx_mem_free(pool_bin);
        vx_mem_free(args_buf);
        vx_dev_close(device);
    }
}


// ------------------------------------------------------------
// MAIN — Pipeline: Conv → ReLU → MaxPool
// ------------------------------------------------------------
int main() {
    RT_CHECK(vx_dev_open(&device));

    const int H = INPUT_SIZE;
    const int W = INPUT_SIZE;

    const int padding = 1;
    const int stride  = 1;

    const int H1 = H;
    const int W1 = W;
    const int C1 = C_OUT;  

    const int H2 = H1/2;
    const int W2 = W1/2;

    // allocate host buffers
    std::vector<float> h_I(C_IN * H * W);
    std::vector<float> h_W(C_OUT * C_IN * K * K);
    std::vector<float> h_O1(C1 * H1 * W1);
    std::vector<float> h_O2(C1 * H2 * W2);
    std::vector<float> h_ref(C1 * H2 * W2);

    // random input
    for (auto& v : h_I) v = rand() / float(RAND_MAX);
    for (auto& v : h_W) v = rand() / float(RAND_MAX);

    // --- Allocate device memory ---
    RT_CHECK(vx_mem_alloc(device, h_I.size()*4, VX_MEM_READ, &I_buf));
    RT_CHECK(vx_mem_alloc(device, h_W.size()*4, VX_MEM_READ, &W_buf));
    RT_CHECK(vx_mem_alloc(device, h_O1.size()*4, VX_MEM_WRITE, &O1_buf));
    RT_CHECK(vx_mem_alloc(device, h_O2.size()*4, VX_MEM_WRITE, &O2_buf));

    uint64_t I_addr, W_addr, O1_addr, O2_addr;
    vx_mem_address(I_buf,  &I_addr);
    vx_mem_address(W_buf,  &W_addr);
    vx_mem_address(O1_buf, &O1_addr);
    vx_mem_address(O2_buf, &O2_addr);

    // upload host data to device
    RT_CHECK(vx_copy_to_dev(I_buf, h_I.data(), 0, h_I.size()*4));
    RT_CHECK(vx_copy_to_dev(W_buf, h_W.data(), 0, h_W.size()*4));

    // load kernel binaries
    RT_CHECK(vx_upload_kernel_file(device, "kernel_conv.vxbin",  &conv_bin));
    RT_CHECK(vx_upload_kernel_file(device, "kernel_relu.vxbin",  &relu_bin));
    RT_CHECK(vx_upload_kernel_file(device, "kernel_pool.vxbin",  &pool_bin));


    // ------------------------------------------------------------
    // 1. Run Convolution
    // ------------------------------------------------------------
    {
        kernel_arg_t arg{};
        arg.I_addr = I_addr;
        arg.W_addr = W_addr;
        arg.O_addr = O1_addr;

        arg.C_in = C_IN;
        arg.C_out = C_OUT;

        arg.height = H;
        arg.width  = W;

        arg.padding = padding;
        arg.stride  = stride;

        arg.H_out = H1;
        arg.W_out = W1;

        arg.grid_dim[0] = W1;
        arg.grid_dim[1] = H1;
        arg.grid_dim[2] = 1;

        RT_CHECK(vx_upload_bytes(device, &arg, sizeof(arg), &args_buf));
        RT_CHECK(vx_start(device, conv_bin, args_buf));
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    }


    // ------------------------------------------------------------
    // 2. Run ReLU (in-place)
    // ------------------------------------------------------------
    {
        kernel_arg_relu_t arg{};
        arg.X_addr = O1_addr;
        arg.total  = C1 * H1 * W1;

        arg.grid_dim[0] = arg.total;

        RT_CHECK(vx_upload_bytes(device, &arg, sizeof(arg), &args_buf));
        RT_CHECK(vx_start(device, relu_bin, args_buf));
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    }


    // ------------------------------------------------------------
    // 3. Run MaxPool → O2
    // ------------------------------------------------------------
    {
        kernel_arg_pool_t arg{};
        arg.I_addr = O1_addr;
        arg.O_addr = O2_addr;

        arg.C = C1;
        arg.H = H1;
        arg.W = W1;

        arg.grid_dim[0] = W2;
        arg.grid_dim[1] = H2;
        arg.grid_dim[2] = C1;

        RT_CHECK(vx_upload_bytes(device, &arg, sizeof(arg), &args_buf));
        RT_CHECK(vx_start(device, pool_bin, args_buf));
        RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    }


    // ------------------------------------------------------------
    // download final output (C1×H2×W2)
    // ------------------------------------------------------------
    RT_CHECK(vx_copy_from_dev(h_O2.data(), O2_buf, 0, h_O2.size()*4));


    // ------------------------------------------------------------
    // CPU reference: conv → relu → pool
    // ------------------------------------------------------------
    std::vector<float> tmp1(C1 * H1 * W1);
    std::vector<float> tmp2(C1 * H2 * W2);

    conv_cpu(tmp1, h_I, h_W, C_IN, C_OUT, H, W, padding, stride);
    relu_cpu(tmp1);
    maxpool_cpu(tmp2, tmp1, C1, H1, W1);

    // compare GPU vs CPU
    int errors = 0;
    for (size_t i = 0; i < tmp2.size(); i++) {
        float a = h_O2[i];
        float b = tmp2[i];

        union { float f; int i; } fa{a}, fb{b};
        if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
            if (errors < 50)
                printf("*** error [%zu] GPU=%f CPU=%f\n", i, a, b);
            errors++;
        }
    }

    if (errors == 0)
        printf("CNN PASSED!\n");
    else
        printf("CNN FAILED with %d errors\n", errors);

    cleanup();
    return errors;
}
