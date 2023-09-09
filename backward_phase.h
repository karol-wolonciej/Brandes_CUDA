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



__global__
__launch_bounds__(SINGLE_WARP_SIZE)
void backward_phase_preparation(int * __restrict__ l,
                                float * __restrict__ time) {             

    *l -= 1;
}



__global__
__launch_bounds__(BLOCK_SIZE_X)
void prepare_delta_for_backward_phase(int * __restrict__ graph_data,
                                      uint32_t * __restrict__ sigma,
                                      double * __restrict__ delta,
                                      float * __restrict__ time) {             

    int threadId = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    
    uint32_t sigma_local;
    int index;

    if (threadId < graph_data[VERTEX_COUNT_ID]) {

        if (threadId < graph_data[VERTEX_COUNT_ID]) {
           
            #pragma unroll
            for (int stream_index = 0; stream_index < NUMBER_OF_STREAMS; stream_index++) {

                index = stream_index * graph_data[PITCHED_VERTEX_COUNT_ID] + threadId;

                sigma_local = sigma[index];
                
                if (sigma_local != 0) {

                    delta[index] = 1 / (double) sigma[index]; 
                }


            }
        }
    }
}



__global__
__launch_bounds__(BLOCK_SIZE_X, DESIRED_MIN_BLOCKS_PER_MULTIPROCESSOR)
void backward_phase(int * __restrict__ GRAPH_R, 
                    int * __restrict__ GRAPH_C,
                    int * __restrict__ OFFSET,
                    int * __restrict__ VMAP,
                    int * __restrict__ NVIR,
                    int * __restrict__ graph_data,
                    int * __restrict__ curr_vertex,
                    int * __restrict__ l,
                    int * __restrict__ d,
                    double * __restrict__ delta,
                    float * __restrict__ time) {                                                                     
    
    int threadId = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;


    int u;
    int virtual_u = threadId;

    int neighbourFirst; 
    int neighbourLast;

    int v;

    int l_local = *l;
    double sum;

    int do_vertex = virtual_u < graph_data[VIRTUAL_VERTEX_COUNT_ID];

    __builtin_expect (do_vertex, true);
    if (do_vertex) {
        u = VMAP[virtual_u];

        neighbourFirst = GRAPH_R[u];
        neighbourLast = GRAPH_R[u + 1];

        if (d[u] == l_local) {
            sum = 0;

            for(int v_id = neighbourFirst + OFFSET[virtual_u]; v_id < neighbourLast; v_id += NVIR[u]) {
                v = GRAPH_C[v_id];

                if (d[v] == l_local + 1) {
                    
                    sum += delta[v];
                }
            }
            atomicAdd(&delta[u], sum);
        }    
    }
}

