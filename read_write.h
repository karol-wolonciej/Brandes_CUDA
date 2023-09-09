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
#include <bits/stdc++.h>
#include <cstdint>

#include "settings.h"



using namespace std;






int getNumberOfLines(string filename) {
    ifstream CountLines(filename);
    string line;
    int lines = 0;

    // Use a while loop together with the getline() function to read the file line by line
    while (getline(CountLines, line)) {
    // Output the text from the file
        lines++;
    }

    // Close the file
    CountLines.close(); 

    return lines;
}

void readGraph(string filename, 
               int &vertex_count,
               int &virtual_vertex_count, 
               uint32_t &edges_count,
               int *& graph_r, 
               int *& graph_c,
               int *& offset, 
               int *& vmap,
               int *& nvir) {

    int lines = getNumberOfLines(filename);
    string line;
    string numbers_line;

    int *left_col = (int*) malloc(lines * sizeof(int)); 
    int *right_col = (int*) malloc(lines * sizeof(int));

    ifstream GraphFile(filename);

    int space_pos;
    string v_left, v_right;
    virtual_vertex_count = 0;

    // int vmap_offset_len = 0;
    int nvir_len = 0;

    int i = 0;
    while (getline (GraphFile, line)) {
        space_pos = line.find(" ");
        v_left = line.substr(0, space_pos);
        v_right = line.substr(space_pos+1);


        left_col[i] = stoi(v_left);
        right_col[i] = stoi(v_right);
        i++;
    }

    GraphFile.close();

    // int first_vertex_id = left_col[0];
    int max_vertex = left_col[0];
    int current_left_vertex;
    int current_right_vertex;
    int bigger_of_two;
    for (int i = 0; i < lines; i++) {
        current_left_vertex = left_col[i];
        current_right_vertex = right_col[i];
        bigger_of_two = (current_left_vertex > current_right_vertex) ? current_left_vertex : current_right_vertex;
        max_vertex = (bigger_of_two > max_vertex) ? bigger_of_two : max_vertex;
    }


    vertex_count = max_vertex + 1;
    nvir_len = vertex_count;

    edges_count = lines * 2;

    int *degree_sort;
    cudaHostAlloc((void**)&degree_sort, (vertex_count) * sizeof(int), cudaHostAllocDefault);


    cudaHostAlloc((void**)&graph_r, (vertex_count + 1) * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&graph_c, edges_count * sizeof(uint32_t), cudaHostAllocDefault);


    memset(graph_r, 0, (vertex_count + 1) * sizeof(int));

    for (int i = 0; i < lines; i++) {
        graph_r[left_col[i] + 1]++;
        graph_r[right_col[i] + 1]++;
    }

    int *graph_r_kolejne = (int*) malloc((vertex_count + 1) * sizeof(int)); 


    graph_r_kolejne[0] = 0;
    for (int i = 1; i < vertex_count + 1; i++) {
        // cout << "sasiedzi " << i << " " << graph_r[i] << endl;
        graph_r[i] += graph_r[i-1];
        graph_r_kolejne[i] = graph_r[i];
    }



    // calculate offset vmap len
    for (int i = 0; i < vertex_count; i++) {
        virtual_vertex_count += ceil((float) (graph_r[i+1] - graph_r[i]) / NUMBER_OF_NEIGBOURS_IN_STRIDE_GRAPH);
    }
    
    offset = (int*) malloc(virtual_vertex_count * sizeof(int)); 
    vmap = (int*) malloc(virtual_vertex_count * sizeof(int)); 
    
    nvir = (int*) malloc(nvir_len * sizeof(int)); 

    // calculate nvir
    for (int i = 0; i < vertex_count; i++) {
        nvir[i] = ceil((float) (graph_r[i+1] - graph_r[i]) / NUMBER_OF_NEIGBOURS_IN_STRIDE_GRAPH);
    }

    int next_vmap_offset_index = 0;
    for (int i = 0; i < vertex_count; i++) {
        int number_of_sub =  ceil((float) (graph_r[i+1] - graph_r[i]) / NUMBER_OF_NEIGBOURS_IN_STRIDE_GRAPH);
        for (int j = 0; j < number_of_sub; j++) {
            offset[next_vmap_offset_index] = j;
            vmap[next_vmap_offset_index] = i;
            next_vmap_offset_index++;
        }
    }

    // calculate adjs/grap_c
    for (int i = 0; i < lines; i++) {
        graph_c[graph_r_kolejne[left_col[i]]] = right_col[i];
        graph_c[graph_r_kolejne[right_col[i]]] = left_col[i];
        graph_r_kolejne[left_col[i]]++;
        graph_r_kolejne[right_col[i]]++;
    }



    free(graph_r_kolejne);
    free(left_col);
    free(right_col);
}






void save_to_file(string filename, int vertex_count, double *BC) {
    ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);  


    for (int i = 0; i < vertex_count; i++) {
        outfile << fixed << BC[i] << endl;  
    }
  
    outfile.close();    
}