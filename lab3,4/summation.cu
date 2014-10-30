#include "utils.h"
#include <stdlib.h>
#include <unistd.h>

struct results
{
	float sum;
};

#include "summation_kernel.cu"

// CPU implementation
float log2_series(int n)
{
        float result = 0;
        for (int i=n-1; i>=0; i--) {
                result = result + (float)((((i%2)-1)+(i%2)))*-1. / (float)(i+1);
        }
	return result;

}



int main(int argc, char ** argv)
{

    if(argc < 3){
        printf("\nYou must specify:  number of blocks, and number of threads per block.\n");
        return 1;
    }   
    int data_size = 1024 * 1024 * 128;

    // Run CPU version
    double start_time = getclock();
    float log2 = log2_series(data_size);
    double end_time = getclock();
    
    printf("CPU result: %f\n", log2);
    printf(" log(2)=%f\n", log(2.0));
    printf(" time=%fs\n", end_time - start_time);
    
    // Parameter definition
    int threads_per_block = 4 * 32;
    int blocks_in_grid = 8;

    blocks_in_grid = atoi(argv[1]);
    threads_per_block = atoi(argv[2]);

    
    int num_threads = threads_per_block * blocks_in_grid;

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    int results_size = num_threads;
    results * data_out_cpu;
    // Allocating output data on CPU
    data_out_cpu = (results *) malloc(results_size*sizeof(results));


    // Allocating output data on GPU
    results *data_out_gpu;
    cudaMalloc((void**)&data_out_gpu,results_size*sizeof(results));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Execute kernel
    summation_kernel<<<blocks_in_grid,threads_per_block>>>(data_size,data_out_gpu);

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get results back
    cudaMemcpy(data_out_cpu,data_out_gpu,results_size*sizeof(results),cudaMemcpyDeviceToHost);
    
    // Finish reduction
    float sum = 0.;
    for (int i = 0; i<results_size; i++) {
	sum+=data_out_cpu[i].sum;
    } 

    // Timer initialization
    cudaEvent_t start_2, stop_2;
    CUDA_SAFE_CALL(cudaEventCreate(&start_2));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_2));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start_2, 0));

    // Execute kernel

    summation_kernel_2<<<blocks_in_grid,threads_per_block>>>(data_size,data_out_gpu);

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop_2, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_2));

    // Get results back
    cudaMemcpy(data_out_cpu,data_out_gpu,results_size*sizeof(results),cudaMemcpyDeviceToHost);
    
    // Finish reduction
    float sum_2 = 0.;
    for (int i = 0; i<results_size; i++) {
	sum_2+=data_out_cpu[i].sum;
    } 
    
    // Cleanup
    cudaFree(data_out_gpu);
    free(data_out_cpu);

    // TODO
    
    printf("GPU results:\n");
    printf(" Sum1: %f\n", sum);
    printf(" Sum2: %f\n", sum_2);
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    float total_time = elapsedTime / 1000.;	// s
    float time_per_iter = total_time / (float)data_size;
    float bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time 1: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);

    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start_2, stop_2));	// In ms

    total_time = elapsedTime / 1000.;	// s
    time_per_iter = total_time / (float)data_size;
    bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time 2: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);
  

    CUDA_SAFE_CALL(cudaEventDestroy(start_2));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_2));
    return 0;
}

