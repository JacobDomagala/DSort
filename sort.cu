#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>
#include <inttypes.h>

#include "curand.h"
#include "curand_kernel.h"
#include <algorithm>

#define ARRAY_SIZE 10000000
#define myCeil(num1,num2) num1 % num2 == 0 ? num1/num2 : 1 + num1/num2
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}

struct Timer {
    int64_t deltaTime = 0; // in milliseconds
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    void ToggleTimer() {
        auto timeNow = std::chrono::steady_clock::now();
        deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow - begin).count();
        begin = timeNow;
    }
};

#pragma region CPU
void STDSort(unsigned int* h_array)
{
	Timer timer;
	timer.ToggleTimer();
	std::sort(h_array, h_array + ARRAY_SIZE - 1);
	timer.ToggleTimer();

	printf("CPU std::sort time: %" PRId64 "ms\n", timer.deltaTime);
}
#pragma endregion

#pragma region GPU


__global__ void MaxReduce(unsigned int* in, size_t numElems)
{
	__shared__ float sMem[1024]; // blockdim

	unsigned int idx = threadIdx.x;
	unsigned int global = idx * blockIdx.x*blockDim.x;

	sMem[idx] = global >= numElems ? 0 : in[idx];
	__syncthreads();

	for (int i = blockDim.x / 2; i > 0; i >>= 1)
	{
		if (idx < i)
		{
			sMem[idx] = max(sMem[idx],sMem[idx + i]);
		}
		__syncthreads();
	}
	if (idx == 0)
	{
		in[blockIdx.x] = sMem[0];
	}
}

__global__ void Scan(unsigned int* const d_list, unsigned int* const d_block_sums, size_t numElems)
{
	extern __shared__ unsigned int s_block_scan[];

	const unsigned int tid = threadIdx.x;
	const unsigned int id = blockDim.x * blockIdx.x + tid;

	// copy to shared memory, pad the block that is too small
	s_block_scan[tid] = id >= numElems ? 0 : d_list[id];
	__syncthreads();

	// reduce
	unsigned int i;
	for (i = 2; i <= blockDim.x; i <<= 1)
	{
		if ((tid + 1) % i == 0)
		{
			unsigned int neighbor_offset = i >> 1;
			s_block_scan[tid] += s_block_scan[tid - neighbor_offset];
		}
		__syncthreads();
	}
	// return i to last value before for loop exited
	i >>= 1;

	// reset last (sum of whole block) to identity element
	if (tid == (blockDim.x - 1))
	{
		d_block_sums[blockIdx.x] = s_block_scan[tid];
		s_block_scan[tid] = 0;
	}
	__syncthreads();

	// downsweep
	for (i = i; i >= 2; i >>= 1)
	{
		if ((tid + 1) % i == 0)
		{
			unsigned int neighbor_offset = i >> 1;
			unsigned int old_neighbor = s_block_scan[tid - neighbor_offset];
			s_block_scan[tid - neighbor_offset] = s_block_scan[tid]; // copy
			s_block_scan[tid] += old_neighbor;
		}
		__syncthreads();
	}

	// copy result to global memory
	if (id < numElems)
	{
		d_list[id] = s_block_scan[tid];
	}
}
__global__ void SumBlocks(unsigned int* const d_predicateScan, unsigned int* const d_blockSumScan)
{
	const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= ARRAY_SIZE)
		return;

	d_predicateScan[id] += d_blockSumScan[blockIdx.x];
}
__global__ void Predicate(unsigned int* d_array, unsigned int* d_compact, char bitCheck)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= ARRAY_SIZE)
		return;

	d_compact[idx] = ((d_array[idx] >> bitCheck) & 1) == 0;
}
__global__ void ShiftBits(unsigned int* input)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= ARRAY_SIZE)
		return;

	input[idx] = (input[idx] + 1) % 2;
}
__global__ void Scatter(unsigned int* read, unsigned int* write, unsigned int* offsetVal, unsigned int* compactShifted,
	unsigned int* compactScanned, unsigned int* compactNegatedScanned)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= ARRAY_SIZE)
		return;

	unsigned int tmpVal;
	if (compactShifted[idx] == 1)
	{
		tmpVal = compactNegatedScanned[idx] + offsetVal[0];
	}
	else
	{
		tmpVal = compactScanned[idx];
	}
	write[tmpVal] = read[idx];
}
__global__ void BitShift(unsigned int* d_array, char bitShift)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= ARRAY_SIZE)
		return;

	d_array[idx] = d_array[idx] >> bitShift;
}
char ComputeBitCount(unsigned int* d_array)
{
	dim3 tmpBlockSize = myCeil(ARRAY_SIZE, 1024);

	// temporary variables
	unsigned int* d_reduced;
	unsigned int h_result;
	cudaMalloc(&d_reduced, sizeof(unsigned int)* ARRAY_SIZE);
	cudaMemcpy(d_reduced, d_array, sizeof(unsigned int)* ARRAY_SIZE, cudaMemcpyDeviceToDevice);

	// reduction on GPU
	MaxReduce <<< tmpBlockSize, 1024 >>>(d_reduced, ARRAY_SIZE);
	cudaDeviceSynchronize();
	while (tmpBlockSize.x > 1024)
	{
		if(tmpBlockSize.x > 1024)
		{
			tmpBlockSize = myCeil(tmpBlockSize.x, 1024);
			MaxReduce << < tmpBlockSize, 1024 >> >(d_reduced, tmpBlockSize.x);
			cudaDeviceSynchronize();
		}
	};

	MaxReduce <<< 1, 1024 >>>(d_reduced, tmpBlockSize.x);
	cudaDeviceSynchronize();

	cudaMemcpy(&h_result, d_reduced, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	unsigned int maxVal = h_result;

	// compute num bits
	char numBits = 0;
	for (int i = 31; i > 0; --i)
	{
		if (((maxVal >> i) & 1) == 1)
		{
			numBits = i+1;
			goto Finish;
		}
	}

Finish:

	// Clean up
	cudaFree(d_reduced);

	return numBits;
}

void RadixSortGPU(unsigned int* d_array)
{
	const unsigned short BLOCK_SIZE = 512;
	dim3 gridSize = myCeil(ARRAY_SIZE, BLOCK_SIZE);
	dim3 tmpGridSize = myCeil(gridSize.x, BLOCK_SIZE);

	unsigned int* d_blockSums;
	unsigned int* d_tmp_Array;
	unsigned int* d_compact;
	unsigned int* d_compactScanned;
	unsigned int* d_zerosOffset;
	unsigned int* d_zerosOffsetArray;
	unsigned int* d_onesOffsetArray;
	unsigned int* d_onesOffset;
	unsigned int* d_compactShiftedtScanned;

	const size_t sizeOfArray = sizeof(unsigned int)* ARRAY_SIZE;
	const size_t elemsOfBlocksums = myCeil(ARRAY_SIZE, BLOCK_SIZE) <= BLOCK_SIZE ? BLOCK_SIZE : myCeil(ARRAY_SIZE, BLOCK_SIZE);
	const size_t sizeOfBlockSums = sizeof(unsigned int)* elemsOfBlocksums;
	const size_t sizeOfSharedMem = sizeof(unsigned int)* BLOCK_SIZE;
	const size_t elemsOfOffset = myCeil(gridSize.x, BLOCK_SIZE);
	const size_t sizeOfOffset = sizeof(unsigned int) * elemsOfOffset;

	char numBits = ComputeBitCount(d_array);

	cudaMalloc(&d_blockSums, sizeOfBlockSums);
	cudaMalloc(&d_tmp_Array, sizeOfArray);
	cudaMalloc(&d_compact, sizeOfArray);
	cudaMalloc(&d_compactScanned, sizeOfArray);
	cudaMalloc(&d_compactShiftedtScanned, sizeOfArray);
	cudaMalloc(&d_zerosOffset, sizeOfOffset);
	cudaMalloc(&d_zerosOffsetArray, sizeOfOffset);
	cudaMalloc(&d_onesOffset, sizeOfOffset);
	cudaMalloc(&d_onesOffsetArray, sizeOfOffset);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	Timer time;
	time.ToggleTimer();

	for (char i = 0; i <= numBits; ++i)
	{
		if (i % 2 == 0)
		{
			Predicate <<< gridSize, BLOCK_SIZE >>>(d_array, d_compact, i);
			cudaDeviceSynchronize();
		}
		else
		{
			Predicate <<< gridSize, BLOCK_SIZE >>>(d_tmp_Array, d_compact, i);
			cudaDeviceSynchronize();
		}
		cudaMemcpy(d_compactScanned, d_compact, sizeOfArray, cudaMemcpyDeviceToDevice);
		cudaMemset(d_blockSums, 0, sizeOfBlockSums);

		// Scan for zeros

		Scan <<< gridSize, BLOCK_SIZE, sizeOfSharedMem >>>(d_compactScanned, d_blockSums, ARRAY_SIZE);
		cudaDeviceSynchronize();

		Scan <<< tmpGridSize, BLOCK_SIZE, sizeOfSharedMem >>>(d_blockSums, d_onesOffsetArray, elemsOfBlocksums);
		cudaDeviceSynchronize();

		Scan <<< 1, BLOCK_SIZE, sizeOfSharedMem >>>(d_onesOffsetArray, d_onesOffset, tmpGridSize.x);
		cudaDeviceSynchronize();

		SumBlocks <<< tmpGridSize, BLOCK_SIZE >>>(d_blockSums, d_onesOffsetArray);
		cudaDeviceSynchronize();

		SumBlocks <<< gridSize, BLOCK_SIZE >>>(d_compactScanned, d_blockSums);
		cudaDeviceSynchronize();

		// scan for ones
		ShiftBits <<< gridSize, BLOCK_SIZE >>>(d_compact);
		cudaDeviceSynchronize();

		cudaMemcpy(d_compactShiftedtScanned, d_compact, sizeOfArray, cudaMemcpyDeviceToDevice);
		cudaMemset(d_blockSums, 0, sizeOfBlockSums);

		Scan <<< gridSize, BLOCK_SIZE, sizeOfSharedMem >>>(d_compactShiftedtScanned, d_blockSums, ARRAY_SIZE);
		cudaDeviceSynchronize();

		Scan <<< tmpGridSize, BLOCK_SIZE, sizeOfSharedMem >>>(d_blockSums, d_zerosOffsetArray, elemsOfBlocksums);
		cudaDeviceSynchronize();

		Scan <<< 1, BLOCK_SIZE, sizeOfSharedMem >>>(d_zerosOffsetArray, d_zerosOffset, tmpGridSize.x);
		cudaDeviceSynchronize();

		SumBlocks <<< tmpGridSize, BLOCK_SIZE >>>(d_blockSums, d_zerosOffsetArray);
		cudaDeviceSynchronize();

		SumBlocks <<< gridSize, BLOCK_SIZE >>>(d_compactShiftedtScanned, d_blockSums);
		cudaDeviceSynchronize();

		if (i % 2 == 0)
		{
			Scatter <<< gridSize, BLOCK_SIZE >>>(d_array, d_tmp_Array, d_onesOffset, d_compact, d_compactScanned, d_compactShiftedtScanned);
			cudaDeviceSynchronize();
		}
		else
		{
			Scatter <<< gridSize, BLOCK_SIZE >>>(d_tmp_Array, d_array, d_onesOffset, d_compact, d_compactScanned, d_compactShiftedtScanned);
			cudaDeviceSynchronize();
		}

	}
	cudaDeviceSynchronize();

	time.ToggleTimer();

	printf("GPU RadixSort time: %" PRId64 "ms\n", time.deltaTime);

	cudaFree(d_blockSums);
	cudaFree(d_tmp_Array);
	cudaFree(d_compact);
	cudaFree(d_compactScanned);
	cudaFree(d_zerosOffset);
	cudaFree(d_onesOffset);
	cudaFree(d_compactShiftedtScanned);
	cudaFree(d_onesOffsetArray);
	cudaFree(d_zerosOffsetArray);
}

#pragma endregion

int main(int argc, char** argv)
{
#pragma region DEVICE SPEC

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf(" Device name: %s\n", prop.name);
	printf(" Compute v%d.%d\n", prop.major, prop.minor);
	printf(" Memory Clock Rate (MHz): %d\n",
		prop.memoryClockRate / 1000);
	printf(" Memory Bus Width (bits): %d\n",
		prop.memoryBusWidth);
	printf(" Total memory (MB): %lu\n",
		prop.totalGlobalMem/(1024*1024));
	printf(" Peak Memory Bandwidth (GB/s): %f\n\n",
		2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8.) / 1.0e6);

#pragma endregion

	// Initialize an array on CPU
	unsigned int* h_array = new unsigned int[ARRAY_SIZE];

	unsigned int* d_array;
	cudaMalloc(&d_array, sizeof(unsigned int)* ARRAY_SIZE);

	curandGenerator_t rng;
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(rng, 0);
	curandGenerate(rng, d_array, ARRAY_SIZE);
	curandDestroyGenerator(rng);

	dim3 gridSize = myCeil(ARRAY_SIZE, 1024);
	BitShift <<< gridSize, 1024 >>>(d_array, 15);
	cudaMemcpy(h_array, d_array, sizeof(unsigned int)*ARRAY_SIZE, cudaMemcpyDeviceToHost);

	RadixSortGPU(d_array);
	STDSort(h_array);

	// clean up
	cudaFree(d_array);
	delete[] h_array;
}
