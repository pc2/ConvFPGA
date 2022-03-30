# ConvFPGA

ConvFPGA is an OpenCL based library for Fast Fourier Transformation based Convolutions on FPGAs.

## Features

- FFT-based 3D Convolution
- Input sizes of powers of 2
- Single Precision (32 bit floating point)
- Batched convolutions
- OpenCL Shared Virtual Memory (SVM) extensions for data transfers

## Supported FPGAs

This library has been tested using the following FPGAs present in the [Noctua](https://pc2.uni-paderborn.de/hpc-services/available-systems/noctua1/) cluster of the Paderborn Center for Parallel Computing (PC2) at Paderborn University:

- [Bittware 520N](https://www.bittware.com/fpga/520n/) card with Intel Stratix 10 GX 2800 FPGA
- [Intel FPGA PAC D5005](https://www.intel.com/content/www/us/en/programmable/products/boards_and_kits/dev-kits/altera/intel-fpga-pac-d5005/overview.html) card with Intel Stratix 10 SX 2800 FPGA

## Quick Setup

Firstly, *dependencies* for building the system
- [CMake](https://cmake.org/) >= 3.16
- C++ compiler with C++11 support (GCC 4.9.0+)
- Intel FPGA SDK for OpenCL
- FFTW3

Once you have this covered, execute the following:

```bash
mkdir build && cd build  # Directory to store build outputs
cmake ..
make
```

You have built the *API* i.e., the OpenCL host code that invokes different transformations correctly are packed into a static library. This must be linked to an application that helps invoke the APIs

*Strictly said*, you have built the following:

- `convfpga` static library, linked such as `-l`
- `convfpga/convfpga.h` header file
- `convolution` - an application which links and includes the above two.

Now onto synthesizing the OpenCL Convolution kernels. These can be synthesized to run on software emulation or on hardware as bitstreams.

- Emulation

```bash
make <kernel_name>_emu
make conv3D_emulate
```

- Hardware Bitstream

```bash
make <kernel_name>_syn
make conv3D_syn
```

Putting them all together, in order to execute the required convolution, set the path to the synthesized bitstream along with other correct configurations as command line parameters to the sample application generated.

```bash
./convolution --num=64 --path=conv3d.aocx
```

*Tip*: for emulation, use the `--emulate` command line parameter.

For explanations regarding the command line options and the OpenCL kernels, check out x and y in the user guide.  

## Related Repositories

- [FFTFPGA](https://github.com/pc2/fft3d-fpga) - an OpenCL based library for Fast Fourier Transformations for FPGAs
- [FFTFPGA-eval](https://git.uni-paderborn.de/arjunr/fftfpga-eval) - archives reports and measurements from FFTFPGA and ConvFPGA

## Contact

- [Arjun Ramaswami](https://github.com/arjunramaswami)
- [Tobias Kenter](https://www.uni-paderborn.de/person/3145/)
- [Thomas D. Kühne](https://chemie.uni-paderborn.de/arbeitskreise/theoretische-chemie/kuehne/)
- [Christian Plessl](https://github.com/plessl)

## Acknowledgements

- [Marius Meyer](https://pc2.uni-paderborn.de/about-pc2/staff-board/staff/person/?tx_upbperson_personsite%5BpersonId%5D=40778&tx_upbperson_personsite%5Bcontroller%5D=Person&cHash=867dec7cae43afd76c85cd503d8da47b) for code reviews, testing and discussions.
