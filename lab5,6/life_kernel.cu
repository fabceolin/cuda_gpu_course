
__global__ void init_kernel(int * domain, int pitch, int block_y_step)
{
                          /* 512 / 4 */

    int tx = threadIdx.x % blockDim.x;
    int ty = (blockIdx.y * blockDim.y) + threadIdx.y;

    int value = tx % 3;
    switch (value) {
        case(0):
            domain[ tx + ty * blockDim.x] = 1;
            break;
        case(1):
            domain[ tx + ty * blockDim.x] = 0;
            break;
        case(2):
            domain[ tx + ty * blockDim.x] = 2;
            break;
    }

}

// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    int domain_x, int domain_y, int pitch)
{
    x = (unsigned)  (x + dx) % domain_x;// Wrap around
    y = (unsigned)  (y + dy) % domain_y;
    return source_domain[y * (pitch / sizeof(int)) + x];
}

__device__ void write_cell(int * dest_domain, int x, int y, int dx, int dy,
    int domain_x, int domain_y, int pitch, int value)
{
    x = (unsigned)(x + dx) % domain_x; // Wrap around
    y = (unsigned)(y + dy) % domain_y;
    dest_domain[y * (pitch / sizeof(int)) + x] = value;
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain, int domain_x, int domain_y, int pitch)
{
    extern __shared__ int shared_data[];

    int init_tx = threadIdx.x * CELLS_PER_THREADS;
            /* 0-511 */
            /*       0-127           */

/*  
    global memory 
    X  --------->                                                                                                                 Y
  00210210210210210210210210210210210210210210210210210210210210210210210210210020100210211002210210210210210210020100020002210210 |
  01000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 |
  11000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 V
  11000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
  */

    int ty = blockIdx.y * blockDim.y + (threadIdx.y);

    int init_shared_tx = init_tx;
    int shared_ty = ty % blockDim.y + 1;

    // load shared;
    /*
 
                                                                                                                                   127
   0                                                                                                                              /
       // Shared memory                                                                                                           |
0  00210210210210210210210210210210210210210210210210210210210210210210210210210020100210211002210210210210210210020100020002210210

   X(shared_x=0, shared_y= 1)  --------->                                                                                         Y
1  00210210210210210210210210210210210210210210210210210210210210210210210210210020100210211002210210210210210210020100020002210210 |
2  01000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 |
3  11000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 V
4  11000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

5  11000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
       */
    for (int i=0 ; i<CELLS_PER_THREADS; i++ ) {
        shared_data[init_shared_tx + i + (shared_ty)*blockDim.x*CELLS_PER_THREADS] = read_cell(source_domain, init_tx, ty, i, 0, domain_x, domain_y, pitch);
        if (shared_ty == 1) {
            shared_data[init_shared_tx + i + (shared_ty-1)*blockDim.x*CELLS_PER_THREADS] = read_cell(source_domain, init_tx, ty, i, -1, domain_x, domain_y, pitch);
        }
        if (shared_ty == 4) {
            shared_data[init_shared_tx + i + (shared_ty+1)*blockDim.x*CELLS_PER_THREADS] = read_cell(source_domain, init_tx, ty, i, 1, domain_x, domain_y, pitch);
        }
    }
    __syncthreads();
    if ( (threadIdx.x == 0) && (threadIdx.y==0) && (blockIdx.y==0 )) {
        int i;
        for (i=0;i<768;i++) {
            write_cell(dest_domain, i%(blockDim.x*CELLS_PER_THREADS), i/(blockDim.x*CELLS_PER_THREADS), 0,0,domain_x,domain_y,pitch,(shared_data[i])%10);
        }
    }
    return;
#if 0
    __syncthreads();
    // Read cell
    int myself = shared_data[shared_tx + (shared_ty)*blockDim.x];


    // TODO: Read the 8 neighbors and count number of blue and red
    int blue=0;
    int red=0;
    int adjacent_count=0;
    for (int i=0; i<9;i++) {
        if (i==4) /* itself */ {
            continue;
        }
        int x = i % 3 - 1;
        int y = (int) (i / 3) - 1;
        int near = shared_data[(((x+shared_tx+blockDim.x)%blockDim.x) + ((shared_ty+y)*blockDim.x))];
        switch (near) {
            case (1):
                red++;
                break;
            case (2):
                blue++;
                break;
            default:
                break;
        }
        adjacent_count = adjacent_count + (!((i+1)%2) && near);
    }

    int total_near = blue+red;
    int new_value = myself;
    // rules
    if ((total_near)>3) {
        new_value = 0;
    }

    if (adjacent_count==1) {
        new_value = 0;
    }

    if ((total_near)==3 && (myself==0)) {
        new_value = 1 << ((blue & 0x02) >> 1);
    }

    write_cell(dest_domain, tx, ty, 0,0,domain_x,domain_y,pitch,new_value);
    return;
#endif
}

