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

#include "forward_phase.h"
#include "backward_phase.h"
#include "updating_bc.h"
#include "read_write.h"

using namespace std;


// σ sigma
// δ delta


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}





int main(int argc, char *argv[]) {
    gpuErrchk( cudaPeekAtLastError() );

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    
    string inputFile(argv[1]);
    string outputFile(argv[2]);

    int vertex_count_cpu;
    int virtual_vertex_count_cpu;
    uint32_t edges_count_cpu;

    int *GRAPH_R_cpu;
    int *GRAPH_C_cpu;
    int *OFFSET_cpu;
    int *VMAP_cpu;
    int *NVIR_cpu;



    readGraph(inputFile, 
              vertex_count_cpu, 
              virtual_vertex_count_cpu,
              edges_count_cpu,
              GRAPH_R_cpu, 
              GRAPH_C_cpu,
              OFFSET_cpu,
              VMAP_cpu,
              NVIR_cpu);



    size_t pitch_vertex_count;
    size_t pitch_virtual_vertex_count;
    size_t pitch_edge_count;

    int* v_arr_pitch;
    int* vv_arr_pitch;
    int* e_arr_pitch;

    //ja jestem swiadomy ze to troche niestandardowy sposob obliczenia dopelnienia do 512...
    cudaMallocPitch(&v_arr_pitch, &pitch_vertex_count, vertex_count_cpu*sizeof(int), 1);
    cudaMallocPitch(&vv_arr_pitch, &pitch_virtual_vertex_count, virtual_vertex_count_cpu*sizeof(int), 1);
    cudaMallocPitch(&e_arr_pitch, &pitch_edge_count, edges_count_cpu*sizeof(int), 1);

    cudaFree(v_arr_pitch);
    cudaFree(vv_arr_pitch);
    cudaFree(e_arr_pitch);


    cudaStream_t streams[NUMBER_OF_STREAMS];
    cudaStream_t streams_memset[NUMBER_OF_STREAMS];
    cudaStream_t streams_memcpy[NUMBER_OF_STREAMS];

    for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < NUMBER_OF_MEMSET_STREAMS; i++) {
        cudaStreamCreate(&streams_memset[i]);
    }

    for (int i = 0; i < NUMBER_OF_MEMCPY_STREAMS; i++) {
        cudaStreamCreate(&streams_memcpy[i]);
    }


    // declare cpu memory
    int *graph_data_cpu; 
    double *global_bc_cpu;
    int *curr_vertex_cpu;
    bool *cont_cpu;
    int *l_cpu;


    float *time_cpu;

    // allocate host memory
    cudaHostAlloc((void**)&graph_data_cpu, GRAPH_DATA_COUNT * sizeof(int), cudaHostAllocDefault);


    cudaHostAlloc((void**)&global_bc_cpu, vertex_count_cpu * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&curr_vertex_cpu, NUMBER_OF_STREAMS * sizeof(int), cudaHostAllocWriteCombined); //write combined
    cudaHostAlloc((void**)&cont_cpu, NUMBER_OF_STREAMS * sizeof(bool), cudaHostAllocDefault);
    cudaHostAlloc((void**)&l_cpu, NUMBER_OF_STREAMS * sizeof(int), cudaHostAllocDefault);

    cudaHostAlloc((void**)&time_cpu, (vertex_count_cpu + 1) * sizeof(float), cudaHostAllocDefault);


    // initialize device memory
    graph_data_cpu[VERTEX_COUNT_ID] = vertex_count_cpu;
    graph_data_cpu[PITCHED_VERTEX_COUNT_ID] = pitch_vertex_count / sizeof(int);
    graph_data_cpu[VIRTUAL_VERTEX_COUNT_ID] = virtual_vertex_count_cpu;
    graph_data_cpu[PITCHED_VIRTUAL_VERTEX_COUNT_ID] = pitch_virtual_vertex_count / sizeof(int);
    graph_data_cpu[EDGES_COUNT_ID] = edges_count_cpu;
    graph_data_cpu[PITCHED_EDGES_COUNT_ID] = pitch_edge_count / sizeof(int);


    // declare gpu memory
    int *GRAPH_R_gpu;
    int *GRAPH_C_gpu;

    int *OFFSET_gpu;
    int *VMAP_gpu;
    
    int *NVIR_gpu;

    
    int *graph_data_gpu; 
    double *global_bc_gpu;
    double *streams_partials_bc_gpu;
    int *streams_d_gpu;
    uint32_t *streams_sigma_gpu;
    double *streams_delta_gpu;

    int *curr_vertex_gpu;
    bool *cont_gpu;
    int *l_gpu;

    float *time_gpu;


    //  allocate device memory
    size_t d_size_row;
    size_t sigma_size_row;

    size_t graph_r_size;
    size_t graph_c_size;

    size_t graph_offset_size;
    size_t graph_vmap_size;
    size_t graph_nvir_size;

    size_t curr_vertex_size;
    size_t cont_vertex_size;    
    size_t l_size;


    cudaMallocPitch(&GRAPH_R_gpu, &graph_r_size, vertex_count_cpu*sizeof(int), 1);
    cudaMallocPitch(&GRAPH_C_gpu, &graph_c_size, edges_count_cpu*sizeof(int), 1);

    cudaMallocPitch(&OFFSET_gpu, &graph_offset_size, virtual_vertex_count_cpu*sizeof(int), 1);
    cudaMallocPitch(&VMAP_gpu, &graph_vmap_size, virtual_vertex_count_cpu*sizeof(int), 1);

    cudaMallocPitch(&NVIR_gpu, &graph_nvir_size, vertex_count_cpu*sizeof(int), 1);

    cudaMalloc((void**)&graph_data_gpu, GRAPH_DATA_COUNT * sizeof(int));

    cudaMallocPitch(&curr_vertex_gpu, &curr_vertex_size, NUMBER_OF_STREAMS*sizeof(int), 1);
    cudaMallocPitch(&cont_gpu, &cont_vertex_size, NUMBER_OF_STREAMS*sizeof(bool), 1);
    cudaMallocPitch(&l_gpu, &l_size, NUMBER_OF_STREAMS*sizeof(int), 1);


    cudaMalloc((void**)&streams_partials_bc_gpu, NUMBER_OF_STREAMS * graph_data_cpu[PITCHED_VERTEX_COUNT_ID] * sizeof(double));
    cudaMalloc((void**)&global_bc_gpu, graph_data_cpu[PITCHED_VERTEX_COUNT_ID] * sizeof(double));
    cudaMalloc((void**)&streams_delta_gpu, NUMBER_OF_STREAMS * graph_data_cpu[PITCHED_VERTEX_COUNT_ID] * sizeof(double));
    
    
    cudaMallocPitch(&streams_d_gpu, &d_size_row, vertex_count_cpu*sizeof(int), NUMBER_OF_STREAMS);
    cudaMallocPitch(&streams_sigma_gpu, &sigma_size_row, vertex_count_cpu*sizeof(uint32_t), NUMBER_OF_STREAMS);



    cudaMalloc((void**)&time_gpu, (vertex_count_cpu + 1) * sizeof(float));

    size_t all_d_size = d_size_row * NUMBER_OF_STREAMS;
    size_t all_sigma_size = sigma_size_row * NUMBER_OF_STREAMS;



    __builtin_assume_aligned(curr_vertex_gpu, 512);
    __builtin_assume_aligned(cont_gpu, 512);
    __builtin_assume_aligned(l_gpu, 512);

    __builtin_assume_aligned(GRAPH_R_gpu, 512);
    __builtin_assume_aligned(GRAPH_C_gpu, 512);

    __builtin_assume_aligned(OFFSET_gpu, 512);
    __builtin_assume_aligned(VMAP_gpu, 512);

    __builtin_assume_aligned(NVIR_gpu, 512);
    
    __builtin_assume_aligned(streams_d_gpu, 512);
    __builtin_assume_aligned(streams_sigma_gpu, 512);
    __builtin_assume_aligned(streams_delta_gpu, 512);


    // initialize and clear device memory
    cudaMemcpy(graph_data_gpu, graph_data_cpu, GRAPH_DATA_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(GRAPH_R_gpu, GRAPH_R_cpu, (vertex_count_cpu + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GRAPH_C_gpu, GRAPH_C_cpu, edges_count_cpu * sizeof(int), cudaMemcpyHostToDevice);



    cudaMemcpy(OFFSET_gpu, OFFSET_cpu, virtual_vertex_count_cpu * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(VMAP_gpu, VMAP_cpu, virtual_vertex_count_cpu * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(NVIR_gpu, NVIR_cpu, vertex_count_cpu * sizeof(int), cudaMemcpyHostToDevice);
   

    cudaMemset(global_bc_gpu, 0, graph_data_cpu[PITCHED_VERTEX_COUNT_ID] * sizeof(double));


    int number_of_blocks_for_stream = ceil((float (graph_data_cpu[VIRTUAL_VERTEX_COUNT_ID])) / BLOCK_SIZE_X);

    int number_of_iterations = ceil((float) vertex_count_cpu / NUMBER_OF_STREAMS); 

    int iteration_global_cont;
    int iteration_global_l;

    int v;



    // make parralel iterations
    int size_of_double_data = NUMBER_OF_STREAMS * graph_data_cpu[PITCHED_VERTEX_COUNT_ID] * sizeof(double);

    for (int it = 0; it < ceil((float) vertex_count_cpu / NUMBER_OF_STREAMS); it++) { //number_of_iterations vertex_count_cpu

        cudaMemsetAsync(streams_partials_bc_gpu, 0, size_of_double_data, streams_memset[0]); //todo zmien 4 na 1 znowu
        cudaMemsetAsync(streams_d_gpu, -1, all_d_size, streams_memset[2]);
        cudaMemsetAsync(streams_sigma_gpu, 0, all_sigma_size, streams_memset[3]);

        cudaMemsetAsync(streams_delta_gpu, 0, size_of_double_data, streams_memset[4]);



        // prepare vertices, conts and l
        for (int i = 0; i < min(vertex_count_cpu - it * NUMBER_OF_STREAMS, NUMBER_OF_STREAMS); i++) {
            v = it * NUMBER_OF_STREAMS + i;
            curr_vertex_cpu[i] = v;
            cont_cpu[i] = false;
            l_cpu[i] = 0;
        }

        cudaMemcpyAsync(curr_vertex_gpu, curr_vertex_cpu, NUMBER_OF_STREAMS * sizeof(int), cudaMemcpyHostToDevice, streams_memcpy[0]);
        cudaMemcpyAsync(cont_gpu, cont_cpu, NUMBER_OF_STREAMS * sizeof(bool), cudaMemcpyHostToDevice, streams_memcpy[1]);
        cudaMemcpyAsync(l_gpu, l_cpu, NUMBER_OF_STREAMS * sizeof(int), cudaMemcpyHostToDevice, streams_memcpy[2]);

        cudaDeviceSynchronize();


        // FORWARD PHASE
        for (int i = 0; i < min(vertex_count_cpu - it * NUMBER_OF_STREAMS, NUMBER_OF_STREAMS); i++) {

            v = it * NUMBER_OF_STREAMS + i;

            cont_cpu[i] = true;
            if (v < vertex_count_cpu) {

                cudaFuncSetAttribute(next_round_preparation, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
                next_round_preparation<<<1, SINGLE_WARP_SIZE, 0, streams[i]>>>(&curr_vertex_gpu[i],
                                                                               &streams_d_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]], 
                                                                               &streams_sigma_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]],
                                                                               time_gpu);



                

            }
            
        }



        iteration_global_cont = true;

        while (iteration_global_cont) {


            iteration_global_cont = false;

            for (int i = 0; i < min(vertex_count_cpu - it * NUMBER_OF_STREAMS, NUMBER_OF_STREAMS); i++) {
                v = it * NUMBER_OF_STREAMS + i;

                if (v < vertex_count_cpu && cont_cpu[i] == true) {


                        cudaFuncSetAttribute(forward_step, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
                        forward_step<<<number_of_blocks_for_stream, BLOCK_SIZE_X, 0, streams[i]>>>(GRAPH_R_gpu, 
                                                                                                   GRAPH_C_gpu,
                                                                                                   OFFSET_gpu,
                                                                                                   VMAP_gpu,
                                                                                                   NVIR_gpu,
                                                                                                   graph_data_gpu,
                                                                                                   &curr_vertex_gpu[i],
                                                                                                   &cont_gpu[i],
                                                                                                   &l_gpu[i], 
                                                                                                   &streams_d_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]], 
                                                                                                   &streams_sigma_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]],
                                                                                                   time_gpu);
                                                                                                   
                }


            }

            cudaDeviceSynchronize();

            cudaMemcpy(cont_cpu, cont_gpu, NUMBER_OF_STREAMS * sizeof(bool), cudaMemcpyDeviceToHost);

            for (int i = 0; i < min(vertex_count_cpu - it * NUMBER_OF_STREAMS, NUMBER_OF_STREAMS); i++) {
                iteration_global_cont = cont_cpu[i] ? true : iteration_global_cont;
            }

            for (int i = 0; i < min(vertex_count_cpu - it * NUMBER_OF_STREAMS, NUMBER_OF_STREAMS); i++) {
                v = it * NUMBER_OF_STREAMS + i;

                if (v < vertex_count_cpu) {

                    cudaFuncSetAttribute(setup_for_next_round, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
                    setup_for_next_round<<<1, SINGLE_WARP_SIZE, 0, streams[i]>>>(&cont_gpu[i],
                                                                                 &l_gpu[i], 
                                                                                 time_gpu);                  
                }
            }
            

        }  

        cudaDeviceSynchronize();



        // BACKWARD PHASE
        cudaMemcpy(l_cpu, l_gpu, NUMBER_OF_STREAMS * sizeof(int), cudaMemcpyDeviceToHost);

        iteration_global_l = 0;
        for (int i = 0; i < min(vertex_count_cpu - it * NUMBER_OF_STREAMS, NUMBER_OF_STREAMS); i++) {
            iteration_global_l = max(iteration_global_l, l_cpu[i]);
        }


        cudaFuncSetAttribute(prepare_delta_for_backward_phase, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
        prepare_delta_for_backward_phase<<<number_of_blocks_for_stream, BLOCK_SIZE_X, 0, 0>>>(graph_data_gpu,
                                                                                              streams_sigma_gpu,
                                                                                              streams_delta_gpu,
                                                                                              time_gpu);


        cudaDeviceSynchronize();

        while (iteration_global_l > 1) {
            iteration_global_l -= 1;

            for (int i = 0; i < min(vertex_count_cpu - it * NUMBER_OF_STREAMS, NUMBER_OF_STREAMS); i++) {
                v = it * NUMBER_OF_STREAMS + i;

                if (v < vertex_count_cpu && l_cpu[i] > 1) {

                        cudaFuncSetAttribute(backward_phase_preparation, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
                        backward_phase_preparation<<<1, SINGLE_WARP_SIZE, 0, streams[i]>>>(&l_gpu[i], 
                                                                                           time_gpu);


                        cudaFuncSetAttribute(backward_phase, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
                        backward_phase<<<number_of_blocks_for_stream, BLOCK_SIZE_X, 0, streams[i]>>>(GRAPH_R_gpu, 
                                                                                                     GRAPH_C_gpu,
                                                                                                     OFFSET_gpu,
                                                                                                     VMAP_gpu,
                                                                                                     NVIR_gpu,
                                                                                                     graph_data_gpu,
                                                                                                     &curr_vertex_gpu[i],
                                                                                                     &l_gpu[i],
                                                                                                     &streams_d_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]], 
                                                                                                     &streams_delta_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]],
                                                                                                     time_gpu);

                                                                                                   

                }
            }

            for (int j = 0; j < NUMBER_OF_STREAMS; j++) {
                l_cpu[j] -= 1;
            }
        }



        for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
                v = it * NUMBER_OF_STREAMS + i;

            // cudaFuncSetAttribute(calculate_partial_bc_values, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
            calculate_partial_bc_values<<<number_of_blocks_for_stream, BLOCK_SIZE_X, 0, streams[i]>>>(graph_data_gpu,
                                                                                                      &curr_vertex_gpu[i],
                                                                                                      &streams_partials_bc_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]],
                                                                                                      &streams_sigma_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]],
                                                                                                      &streams_delta_gpu[i * graph_data_cpu[PITCHED_VERTEX_COUNT_ID]],
                                                                                                      time_gpu);




        }

        

        cudaDeviceSynchronize();


        cudaFuncSetAttribute(update_bc_global_values, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
        update_bc_global_values<<<number_of_blocks_for_stream, BLOCK_SIZE_X, 0, 0>>>(graph_data_gpu,
                                                                                     global_bc_gpu,
                                                                                     streams_partials_bc_gpu,
                                                                                     time_gpu);



        cudaDeviceSynchronize();
    }


    cudaMemcpy(global_bc_cpu, global_bc_gpu, graph_data_cpu[VERTEX_COUNT_ID] * sizeof(double), cudaMemcpyDeviceToHost);


    save_to_file(outputFile, graph_data_cpu[VERTEX_COUNT_ID], global_bc_cpu);


    cudaFree(GRAPH_R_gpu);
    cudaFree(GRAPH_C_gpu);
    cudaFree(OFFSET_gpu);
    cudaFree(VMAP_gpu);
    cudaFree(NVIR_gpu);
    cudaFree(graph_data_gpu); 
    cudaFree(global_bc_gpu);
    cudaFree(streams_partials_bc_gpu);
    cudaFree(streams_d_gpu);
    cudaFree(streams_sigma_gpu);
    cudaFree(streams_delta_gpu);
    cudaFree(curr_vertex_gpu);
    cudaFree(cont_gpu);
    cudaFree(l_gpu);
    cudaFree(time_gpu);

    return 0;
}






