cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("dali-examples")




# additional cmake find files
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# include gflags and make sure namespace is correct depending
# on installed version (google:: VS gflags::)
find_package(Gflags REQUIRED)
add_definitions(-DGFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})
include_directories(${GFLAGS_INCLUDE_DIRS})

find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf)
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/SQLiteCpp)
include_directories(${PROJECT_SOURCE_DIR}/third_party/SQLiteCpp/include)




# ensure we are using C++11
# -stdlib=libc++
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -O3 -g -fPIC" )













set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)
set(EXAMPLES_BUILD_DIR ${PROJECT_SOURCE_DIR}/build/examples)
set(EXAMPLE_PROGRAMS api_showcase
                     arithmetic
                     babi_solvers
                     beam_tree_training
                     bidirectional_sentiment
                     character_prediction
                     grid_search_simple
                     language_model
                     language_model_from_senti
                     lattice_prediction
                     lattice_prediction_from_protobuff
                     loading_from_sqlite
                     lstm_sentiment
                     machine_comprehension
                     mlbasics_learn_to_add
                     mlbasics_rnn_binary_addition
                     sparse_lstm_sentiment
                     sparse_ner
                     sparse_paraphrase
                     visualizer
                     )


add_library(utils examples/utils.cpp examples/utils.h)
target_link_libraries(utils SQLiteCpp sqlite3 proto ${CMAKE_DL_LIBS})



#################### CONVERT ME TO FINDDALI.cmake #############################
# also create DALI_INCLUDES variable
find_package(ZLIB REQUIRED)
find_library(DALI_CORE_LIBRARIES dali)
find_library(DALI_GPU_LIBRARIES dali_cuda)
LIST(APPEND DALI_LIBRARIES ${DALI_CORE_LIBRARIES})
LIST(APPEND DALI_LIBRARIES ${DALI_GPU_LIBRARIES})
###############################################################################


foreach(program ${EXAMPLE_PROGRAMS})
    add_executable(${program} ${EXAMPLES_DIR}/${program}.cpp)
    target_link_libraries(${program} ${DALI_LIBRARIES} ${GFLAGS_LIBRARIES} ${ZLIB_LIBRARIES} utils)
    if (MKL_FOUND AND BLAS_FOUND)
        add_custom_command(TARGET ${program}
            COMMAND sh ${PROJECT_SOURCE_DIR}/scripts/fix_dylib.sh ${EXAMPLES_BUILD_DIR}/${program}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
    endif (MKL_FOUND AND BLAS_FOUND)
endforeach(program)
