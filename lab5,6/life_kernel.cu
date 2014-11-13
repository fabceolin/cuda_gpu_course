
__global__ void init_kernel(int * domain, int pitch)
{
    domain[blockIdx.y * pitch / sizeof(int) + blockIdx.x * blockDim.x + threadIdx.x]
        = (1664525ul * (blockIdx.x + threadIdx.y + threadIdx.x) + 1013904223ul) % 3;
}

// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy,
    int domain_x, int domain_y, int pitch)
{
    x = (x + dx + domain_x) % domain_x;	// Wrap around
    y = (y + dy + domain_y) % domain_y;
    return source_domain[y * (pitch / sizeof(int)) + x];
}

__device__ void write_cell(int * dest_domain, int x, int y, int dx, int dy,
    int domain_x, int domain_y, int pitch, int value)
{
    x = (x + dx + domain_x) % domain_x;	// Wrap around
    y = (y + dy + domain_y) % domain_y;
    dest_domain[y * (pitch / sizeof(int)) + x] = value;
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain,
    int domain_x, int domain_y, int pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
    
    // Read cell
    int myself = read_cell(source_domain, tx, ty, 0, 0,
                       domain_x, domain_y, pitch);
    
    
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
        int near = read_cell(source_domain, tx, ty, x,y,domain_x,domain_y,pitch);
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

