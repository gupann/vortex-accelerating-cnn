// main_conv_layer.cpp
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include <cstdlib>

#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

// -----------------------------------------------------------------------------
// Configuration: 
// -----------------------------------------------------------------------------

static const int C_IN  = 3;   // number of input channels
static const int C_OUT = 4;   // number of output channels
static const int K     = 3;   // kernel size

// -----------------------------------------------------------------------------
// Comparators
// -----------------------------------------------------------------------------

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char* type_str() { return "integer"; }
  static int generate() { return rand(); }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
public:
  static const char* type_str() { return "float"; }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

// -----------------------------------------------------------------------------
// CPU reference implementation: multi-channel 3x3 conv, N=1
// I: [C_in][H][W], W: [C_out][C_in][3][3], O: [C_out][H_out][W_out]
// -----------------------------------------------------------------------------

static void convolution_cpu(TYPE *O,
                            const TYPE *I,
                            const TYPE *W,
                            int32_t C_in,
                            int32_t C_out,
                            int32_t H,
                            int32_t Wt,
                            int32_t padding,
                            int32_t stride) {
  const int K = 3;
  int H_out = (H + 2*padding - K) / stride + 1;
  int W_out = (Wt + 2*padding - K) / stride + 1;

  for (int oc = 0; oc < C_out; ++oc) {
    for (int oy = 0; oy < H_out; ++oy) {
      for (int ox = 0; ox < W_out; ++ox) {
        float sum = 0.0f;

        for (int ic = 0; ic < C_in; ++ic) {
          for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
              int in_y = oy * stride + ky - padding;
              int in_x = ox * stride + kx - padding;

              if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < Wt) {
                int in_idx = ic * H * Wt + in_y * Wt + in_x;
                int wt_idx = ((oc * C_in + ic) * K + ky) * K + kx;
                sum += I[in_idx] * W[wt_idx];
              }
            }
          }
        }

        int out_idx = oc * H_out * W_out + oy * W_out + ox;
        O[out_idx] = static_cast<TYPE>(sum);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Globals for device resources
// -----------------------------------------------------------------------------

const char* kernel_file = "kernel.vxbin";  // compiled from kernel.cpp
int size = 32;                             // input height=width
bool use_lmem = false;

vx_device_h device = nullptr;
vx_buffer_h I_buffer = nullptr;
vx_buffer_h W_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Conv Layer Test." << std::endl;
  std::cout << "Usage: [-k kernel] [-l: local memory] [-n size] [-h|?: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:lh")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'l':
      use_lmem = true;
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(I_buffer);
    vx_mem_free(W_buffer);
    vx_mem_free(O_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "input size: " << size << "x" << size
            << ", C_in=" << C_IN
            << ", C_out=" << C_OUT << std::endl;

  // convolution parameters
  int padding = 1;
  int stride  = 1;
  int H = size;
  int Wt = size;

  int H_out = (H + 2*padding - K) / stride + 1;
  int W_out = (Wt + 2*padding - K) / stride + 1;

  // Setup kernel_arg
  kernel_arg.C_in   = C_IN;
  kernel_arg.C_out  = C_OUT;
  kernel_arg.height = H;
  kernel_arg.width  = Wt;
  kernel_arg.padding = padding;
  kernel_arg.stride  = stride;
  kernel_arg.H_out   = H_out;
  kernel_arg.W_out   = W_out;
  kernel_arg.use_lmem = use_lmem;

  // grid_dim = [W_out, H_out, C_out]
  kernel_arg.grid_dim[0] = W_out;
  kernel_arg.grid_dim[1] = H_out;
  kernel_arg.grid_dim[2] = C_OUT;
  kernel_arg.block_dim[0] = 1;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.block_dim[2] = 1;

  uint32_t i_points = C_IN  * H     * Wt;
  uint32_t w_points = C_OUT * C_IN * K * K;
  uint32_t o_points = C_OUT * H_out * W_out;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  size_t i_nbytes = i_points * sizeof(TYPE);
  size_t w_nbytes = w_points * sizeof(TYPE);
  size_t o_nbytes = o_points * sizeof(TYPE);

  RT_CHECK(vx_mem_alloc(device, i_nbytes, VX_MEM_READ,  &I_buffer));
  RT_CHECK(vx_mem_address(I_buffer, &kernel_arg.I_addr));

  RT_CHECK(vx_mem_alloc(device, w_nbytes, VX_MEM_READ,  &W_buffer));
  RT_CHECK(vx_mem_address(W_buffer, &kernel_arg.W_addr));

  RT_CHECK(vx_mem_alloc(device, o_nbytes, VX_MEM_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));

  if (use_lmem) {
    uint64_t dev_local_mem_size;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_LOCAL_MEM_SIZE, &dev_local_mem_size));
    if (w_nbytes > dev_local_mem_size) {
      std::cout << "Error: Not enough local memory: needed="
                << w_nbytes << ", available=" << dev_local_mem_size << std::endl;
      cleanup();
      return 1;
    }
  }

  std::cout << "dev_argI=0x" << std::hex << kernel_arg.I_addr << std::endl;
  std::cout << "dev_argW=0x" << std::hex << kernel_arg.W_addr << std::endl;
  std::cout << "dev_argO=0x" << std::hex << kernel_arg.O_addr << std::endl;

  // Host buffers
  std::vector<TYPE> h_I(i_points);
  std::vector<TYPE> h_W(w_points);
  std::vector<TYPE> h_O(o_points);

  // Generate random input: I[c, y, x]
  for (int ic = 0; ic < C_IN; ++ic) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < Wt; ++x) {
        int idx = ic * H * Wt + y * Wt + x;
        h_I[idx] = static_cast<TYPE>(rand()) / RAND_MAX;
      }
    }
  }

  // Generate random weights: W[oc, ic, ky, kx]
  for (int oc = 0; oc < C_OUT; ++oc) {
    for (int ic = 0; ic < C_IN; ++ic) {
      for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
          int idx = ((oc * C_IN + ic) * K + ky) * K + kx;
          h_W[idx] = static_cast<TYPE>(rand()) / RAND_MAX;
        }
      }
    }
  }

  // upload input buffer
  {
    std::cout << "upload source buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(I_buffer, h_I.data(), 0, i_nbytes));
  }

  // upload weight buffer
  {
    std::cout << "upload weight buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(W_buffer, h_W.data(), 0, w_nbytes));
  }

  // Upload kernel binary
  std::cout << "Upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));
  printf("HOST sizeof(kernel_arg_t) = %zu\n", sizeof(kernel_arg_t));


  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, o_nbytes));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(o_points);
    convolution_cpu(h_ref.data(), h_I.data(), h_W.data(),
                    C_IN, C_OUT, H, Wt, padding, stride);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      auto ref = h_ref[i];
      auto cur = h_O[i];
      if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
        ++errors;
      }
    }
    std::cout << "Total testing " << std::dec << h_ref.size() << " elements" << std::endl;
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
