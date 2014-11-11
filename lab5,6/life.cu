
#include "utils.h"
#include <stdlib.h>

#include "life_kernel.cu"


int main(int argc, char ** argv)
{
    // Definition of parameters
    int domain_x = 128;// Multiple of threads_per_block * cell_per_word
    int domain_y = 128;

    int cells_per_word = 1;

    int steps = 2;


    int blocks_y_step = 4;
    int threads_per_block = 128 * blocks_y_step;
    int blocks_x = blocks_y_step * domain_x / (threads_per_block * cells_per_word);
    int blocks_y = domain_y / blocks_y_step;

    dim3  grid(blocks_x  , blocks_y );// CUDA grid dimensions
    dim3  threads(threads_per_block);// CUDA block dimensions

    // Allocation of arrays
    int * domain_gpu[2] = {NULL, NULL};

    size_t pitch;
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[0], &pitch,
                domain_x / cells_per_word * sizeof(int),
                domain_y));
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&domain_gpu[1], &pitch,
                domain_x / cells_per_word * sizeof(int),
                domain_y));

    // Arrays of dimensions pitch * domain.y
    init_kernel<<< grid, threads, 0 >>>(domain_gpu[0], pitch);

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Kernel execution
    int shared_mem_size = domain_x * (blocks_y_step+2) * sizeof(int) ;
    printf("%d %d %d \n",blocks_x, blocks_y, shared_mem_size);
    for(int i = 0; i < steps; i++) {
        life_kernel<<< grid, threads, shared_mem_size >>>(domain_gpu[i%2], domain_gpu[(i+1)%2], domain_x, domain_y, pitch, blocks_y_step);
    }

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms
    printf("GPU time: %f ms\n", elapsedTime);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));

    // Get results back
    int * domain_cpu = (int*)malloc(pitch * domain_y);
    CUDA_SAFE_CALL(cudaMemcpy(domain_cpu, domain_gpu[steps%2], pitch * domain_y, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(domain_gpu[0]));
    CUDA_SAFE_CALL(cudaFree(domain_gpu[1]));


    // Count colors
    int red = 0;
    int blue = 0;
    for(int y = 0; y < domain_y; y++)
    {
        for(int x = 0; x < domain_x; x++)
        {
            int cell = domain_cpu[y * pitch/sizeof(int) + x];
            printf("%u", cell);
            if(cell == 1) {
                red++;
            }
            else if(cell == 2) {
                blue++;
            }
        }
        printf("\n");
    }

    printf("Red/Blue cells: %d/%d\n", red, blue);

    free(domain_cpu);

    return 0;
}

