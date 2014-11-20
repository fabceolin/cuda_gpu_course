
#define CELL_INIT_PATTERN 0x49249249

__host__ __device__ unsigned int cellValueDecode(unsigned int cell, unsigned int count) {
    unsigned int mask = 3 << (2*count);
    unsigned int filter = cell & mask;
    return (filter >> 2*count);
}

__host__ __device__ unsigned int cellValueEncode(unsigned int cell, unsigned int count, unsigned int value) {
    unsigned int mask = 3 << 2*count;
    unsigned int inv_mask = ~mask;
    unsigned int filter = cell & inv_mask;
    unsigned int shift_value = value << 2*count;
    cell = filter | shift_value;
    return cell;
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

__device__ unsigned int rotl(unsigned int value, int shift) {
        return (value << shift) | (value >> (sizeof(value) * CHAR_BIT - shift));
}

__device__ int rotr(unsigned int value, int shift) {
        return (value >> shift) | (value << (sizeof(value) * CHAR_BIT - shift));
}

__global__ void init_kernel(int * domain, int domain_x, int domain_y, int pitch)
{
                          /* 512 / 4 */
              /* 0-31        4          0-3 */
    int ty = (blockIdx.y * blockDim.y) + threadIdx.y;
                                     /* 010010 repeated 5 with 1 concatenated */
            /* 0 -127 * 8 + 0-7 */
    unsigned shift = (threadIdx.x % 3);
    unsigned value = rotl(CELL_INIT_PATTERN,shift);
    write_cell(domain, threadIdx.x, ty, 0 , 0 , domain_x, domain_y, pitch, value);
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain, int domain_x, int domain_y, int pitch)
{
    extern __shared__ int shared_data[];

    int tx = threadIdx.x;
             /* 0 - 31 * 4 + 0 - 3 */
    int ty = blockIdx.y * blockDim.y + (threadIdx.y);

    int shared_tx = tx;
    int shared_ty = ty % blockDim.y + 1;

    shared_data[shared_tx + (shared_ty)*blockDim.x] = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y, pitch);
    if (shared_ty == 1) {
        shared_data[shared_tx + (shared_ty-1)*blockDim.x] = read_cell(source_domain, tx, ty, 0, -1, domain_x, domain_y, pitch);
    }
    if (shared_ty == 4) {
        shared_data[shared_tx + (shared_ty+1)*blockDim.x] = read_cell(source_domain, tx, ty, 0, 1, domain_x, domain_y, pitch);
    }

#if 0
    __syncthreads();
    if ( (threadIdx.x == 0) && (threadIdx.y==0) && (blockIdx.y==0 )) {
        int i;
        for (i=0;i<768;i++) {
            write_cell(dest_domain, i%(blockDim.x*CELLS_PER_THREADS), i/(blockDim.x*CELLS_PER_THREADS), 0,0,domain_x,domain_y,pitch,(shared_data[i])%10);
        }
    }
    return;
#endif
    __syncthreads();
    // Read cell

        int myself = cellValueDecode(shared_data[shared_tx + (shared_ty)*blockDim.x],4);


        // TODO: Read the 8 neighbors and count number of blue and red
        int blue=0;
        int red=0;
        int adjacent_count=0;
        for (int i=0; i<9;i++) {
            if (i==4) /* itself */ {
                continue;
            }
            int dx = i % 3 - 1;
            int dy = (int) (i / 3) - 1;
            int near = cellValueDecode(shared_data[((shared_tx+dx)%(blockDim.x)) + ((shared_ty+dy)*blockDim.x)],i);
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
        unsigned int new_cell = cellValueEncode(myself,4,new_value);
        write_cell(dest_domain, tx, ty, 0,0,domain_x,domain_y,pitch,new_cell);
    return;
}

