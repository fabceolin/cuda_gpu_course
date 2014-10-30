
// GPU kernel
__global__ void summation_kernel(int data_size, results * data_out)
{
	int num_threads = gridDim.x * blockDim.x;
	int thread_data_size = data_size / num_threads;
	int thread_absolute_id = blockIdx.x * blockDim.x + threadIdx.x;

        float result = 0;
        for (int i = thread_absolute_id*thread_data_size; i<thread_absolute_id*thread_data_size+thread_data_size; i++) {
                result = result + (float)((((i%2)-1)+(i%2)))*-1. / (float)(i+1);
        }
	data_out[thread_absolute_id].sum = result;
}


__global__ void summation_kernel_2(int data_size, results * data_out)
{

	int num_threads = gridDim.x * blockDim.x;
	int thread_data_step = data_size / num_threads;
	int thread_absolute_id = blockIdx.x * blockDim.x + threadIdx.x;


        float result = 0;
	int i;
	for(int j = 0; j < thread_data_step; j++){
        	i = j * num_threads + thread_absolute_id;
                result = result + (float)((((i%2)-1)+(i%2)))*-1. / (float)(i+1);
        }
	data_out[thread_absolute_id].sum = result;
}
