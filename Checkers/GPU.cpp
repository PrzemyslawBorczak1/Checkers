#include "GPU.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "mcts_kernel.cuh"

GPU* GPU::instance_ = nullptr;

static void exitError(const char* msg, cudaError_t cs) {
	std::fprintf(stderr, "%s :%s\n", msg, cudaGetErrorString(cs));
	std::exit(EXIT_FAILURE);
}

void GPU::gpuPrelude(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]) {
	cudaError_t cs;

	cs = mctsSetSymbols(Neighbours, Captures, Rays);
	if (cs != cudaSuccess) {
		exitError("mctsSetSymbols failed!\n", cs);
	}

	cs = cudaMalloc((void**)&dev_board, sizeof(Board));
	if (cs != cudaSuccess) {
		exitError("cudaMalloc dev_board failed!\n", cs);
	}

}


GPU::GPU(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]) {
	gpuPrelude(Neighbours, Captures, Rays);
}


GPU& GPU::getInstance(int8_t(&Neighbours)[32][4], int8_t(&Captures)[32][4], uint32_t(&Rays)[32][4]) {
	if (!instance_) {
		instance_ = new GPU(Neighbours, Captures, Rays);
	}

	return *instance_;
}

int GPU::simulate(Board board, Color color, uint32_t seed, int moves_without_progress) {
	cudaError_t cs = cudaSuccess;
	cs = cudaMemcpy(dev_board, &board, sizeof(Board), cudaMemcpyHostToDevice);
	if (cs != cudaSuccess) {
		exitError("cudaMemcpy dev_board failed!\n", cs);
	}
	char* d_ret = nullptr;

	uint32_t ret = runMCTS(dev_board, color, seed, moves_without_progress);
	return ret;
}

