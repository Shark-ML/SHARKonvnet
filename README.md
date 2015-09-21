# SHARKonvnet
GPU CNN's for SHARK.

## Installation
* Cuda Toolkit https://developer.nvidia.com/cuda-downloads
* cuDNN https://developer.nvidia.com/cudnn
* SHARK http://image.diku.dk/shark/

## Makefile
* Here be hints on how to make it compile.

## MNIST
* Change directory to `scripts` and run `python prep_mnist_data.py`. This will download, unpack and normalize the MNIST dataset to `data/mnist/`.
* Change directory to the root of the project and run `make mnist`. This will compile `tests/MnistTest.cpp` into `obj/MnistTest.o`.
* Run with `./obj/MnistTest.o`. You should see some info about the convnet (specified in `tests/MnistTest.cpp`), and then some progress info every 500 itertation. After 2000 iterations the 01-loss should be around 4%. It converges to a little below 1%.
