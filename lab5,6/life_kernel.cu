
__global__ void init_kernel(int * domain, int pitch, int block_y_step)
{
                          /* 512 / 4 */

    int blockXThreadSize = blockDim.x / block_y_step;
    int blockYThreadSize = blockDim.x / block_y_step / gridDim.y;

    int tx = threadIdx.x % blockXThreadSize ;
    int ty = (blockIdx.y * blockYThreadSize) + (int)(threadIdx.x / blockXThreadSize);


    domain[ tx + ty * blockXThreadSize] = ( tx + ty * blockXThreadSize) % 3;
}

// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    int domain_x, int domain_y, int pitch)
{
    x = (x + dx) % domain_x;	// Wrap around
    y = (y + dy) % domain_y;
    return source_domain[y * (pitch / sizeof(int)) + x];
}

__device__ void write_cell(int * dest_domain, int x, int y, int dx, int dy,
    int domain_x, int domain_y, int pitch, int value)
{
    x = (x + dx) % domain_x;	// Wrap around
    y = (y + dy) % domain_y;
    dest_domain[y * (pitch / sizeof(int)) + x] = value;
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain, int domain_x, int domain_y, int pitch, int block_y_step)
{
    extern __shared__ int shared_data[];

    int blockXThreadSize = blockDim.x / block_y_step;
    int blockYThreadSize = blockDim.x / block_y_step / gridDim.y;

    int tx = threadIdx.x % blockXThreadSize ;
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
             /* 0-31  */
             /*      0-124 step 4      +         0 - 3             */
            /*                      0-127                          */
    int ty = blockIdx.y * blockYThreadSize + (int)(threadIdx.x / blockXThreadSize);

    /* 0 -127 */
    int shared_tx = tx;

    /* 1 - 4 */
    int shared_ty = (ty % blockYThreadSize) + 1;

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
                /* 0-127  +   (1-4)*128       */
    shared_data[shared_tx + (shared_ty)*blockXThreadSize ] = read_cell(source_domain, tx, ty, 0, 0,
                       domain_x, domain_y, pitch);

    if (shared_ty == 1 ) {
                   /* 0-127   +   0 */
        shared_data[shared_tx + (shared_ty-1)*blockXThreadSize ] = read_cell(source_domain, tx, ty, 0, -1,
                       domain_x, domain_y, pitch);
    }

    if (shared_ty == 4 ) {
                   /* 0-127   +  5*blockDim.x  */
        shared_data[shared_tx + (shared_ty+1)*blockXThreadSize ] = read_cell(source_domain, tx, ty, 0, 1,
                       domain_x, domain_y, pitch);
    }

    __syncthreads();

    // Read cell
//    int myself=0;
    int myself = shared_data[shared_tx + (shared_ty)*blockXThreadSize];


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
  //      int near=0;
        // Modulus of negative is negative, so I will sum 2 to the value to never be negative
        if ((x+shared_tx<0)) {
            x=0;
            shared_tx=blockXThreadSize-1;
        }

        int near = shared_data[(((x+shared_tx)%blockXThreadSize) + ((shared_ty+y)*blockXThreadSize))];
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
        if ( (i+1)%2==0) {
            if (near>0) {
                adjacent_count++;
            }
        }
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
         if (blue>red) {
             new_value=2;
         }
         else {
             new_value=1;
         }
    }

    write_cell(dest_domain, tx, ty, 0,0,domain_x,domain_y,pitch,new_value);
    return;

}

