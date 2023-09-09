#define SINGLE_WARP_SIZE                        32
#define BLOCK_SIZE_X                            256

#define DESIRED_MIN_BLOCKS_PER_MULTIPROCESSOR   8


#define NUMBER_OF_STREAMS                       50
#define NUMBER_OF_MEMSET_STREAMS                5
#define NUMBER_OF_MEMCPY_STREAMS                5

#define NUMBER_OF_NEIGBOURS_IN_STRIDE_GRAPH     8

#define GRAPH_DATA_COUNT                        5

#define VERTEX_COUNT_ID                         0
#define VIRTUAL_VERTEX_COUNT_ID                 1
#define PITCHED_VIRTUAL_VERTEX_COUNT_ID         2
#define PITCHED_VERTEX_COUNT_ID                 3
#define EDGES_COUNT_ID                          4
#define PITCHED_EDGES_COUNT_ID                  5