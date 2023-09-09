#include <time.h>
#include <stdio.h>
#include <iostream>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdint>
#include <numeric>
#include <cuda.h>
#include <fstream>
#include <math.h>
#include <cstdint>

#include "settings.h"


using namespace std;


// σ sigma
// δ delta




// single warp
__global__ 
__launch_bounds__(SINGLE_WARP_SIZE)
void next_round_preparation(int * __restrict__  curr_vertex,
                            int * __restrict__ d,
                            uint32_t * __restrict__ sigma,
                            float * __restrict__ time) {
    d[*curr_vertex] = 0;
    sigma[*curr_vertex] = 1;
}



__global__ 
__launch_bounds__(BLOCK_SIZE_X, DESIRED_MIN_BLOCKS_PER_MULTIPROCESSOR)
void forward_step(int * __restrict__ GRAPH_R, 
                  int * __restrict__ GRAPH_C,
                  int * __restrict__ OFFSET,
                  int * __restrict__ VMAP,
                  int * __restrict__ NVIR,
                  int * __restrict__ graph_data,
                  int * __restrict__ curr_vertex,
                  bool * __restrict__ cont,
                  int * __restrict__ l, 
                  int * __restrict__ d, 
                  uint32_t * __restrict__ sigma,
                  float * __restrict__ time) {                                                                     
                    

    int threadId = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

    int u;
    int virtual_u = threadId;

    int v;
    int d_v;
    
    int neighbourFirst;
    int neighbourLast;

    bool cont_local = false;
    int local_l = *l;

    int do_vertex = virtual_u < graph_data[VIRTUAL_VERTEX_COUNT_ID];

    __builtin_expect (do_vertex, true);
    if (do_vertex) {
        u = VMAP[virtual_u];

        neighbourFirst = GRAPH_R[u];
        neighbourLast = GRAPH_R[u + 1];

        if (d[u] == local_l) {
            for(int v_id = neighbourFirst + OFFSET[virtual_u]; v_id < neighbourLast; v_id += NVIR[u]) {

                v = GRAPH_C[v_id];
                d_v = d[v];
                if (d_v == -1) {
                    d[v] = local_l + 1;
                    cont_local = true;
                }

                if (d[v] == local_l + 1) {
                    atomicAdd(&sigma[v], sigma[u]);
                }

            }

            if (cont_local == true) {
                *cont = true; 
            }
        }
    }
}






// single warp
__global__ 
__launch_bounds__(SINGLE_WARP_SIZE)
void setup_for_next_round(bool * __restrict__ cont,
                          int * __restrict__ l, 
                          float * __restrict__ time) {
    *l += 1;
    *cont = false;
}  

