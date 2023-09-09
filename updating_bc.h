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



__global__ 
__launch_bounds__(BLOCK_SIZE_X, DESIRED_MIN_BLOCKS_PER_MULTIPROCESSOR)
void update_bc_global_values(int * __restrict__ graph_data,
                             double * __restrict__ global_bc,
                             double * __restrict__ partials_bc,
                             float * __restrict__ time) {

    int threadId = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

    double bc_sum_elem = 0;

    int vertex_count = graph_data[VERTEX_COUNT_ID];

    int pitched_vertex_count = graph_data[PITCHED_VERTEX_COUNT_ID];


    int do_vertex = threadId < vertex_count;

    __builtin_expect (do_vertex, true);
    if (threadId < vertex_count) {

        #pragma unroll
        for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
            bc_sum_elem += partials_bc[i * pitched_vertex_count + threadId];
        }

        global_bc[threadId] += bc_sum_elem;

    } 
}



__global__ 
__launch_bounds__(BLOCK_SIZE_X)
void calculate_partial_bc_values(int * __restrict__ graph_data,
                                 int * __restrict__ curr_vertex,
                                 double * __restrict__ partial_bc,
                                 uint32_t * __restrict__ sigma,
                                 double * __restrict__ delta,
                                 float * __restrict__ time) {

    int threadId = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

    int v = threadId;

    double added = 0;

    if (threadId < graph_data[VERTEX_COUNT_ID]) {
        if (v != *curr_vertex) {
            added = delta[v] * sigma[v] - 1;
            partial_bc[v] += added > 0 ? added : 0;
        }
    }

}