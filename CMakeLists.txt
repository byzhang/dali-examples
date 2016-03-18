cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("dali-examples")

# additional cmake find files
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# include gflags and make sure namespace is correct depending
# on installed version (google:: VS gflags::)
find_package(Gflags REQUIRED)
add_definitions(-DGFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})













# CUDA BEGINS HERE
if (DEFINED WITH_CUDA)
    message("WITH_CUDA ${WITH_CUDA}")
else()
    if (CUDA_FOUND STREQUAL TRUE)
        SET(WITH_CUDA TRUE)
    else()
        SET(WITH_CUDA FALSE)
        message("CUDA not found - did you know that Dali has GPU support?")
    endif()
endif(DEFINED WITH_CUDA)

if (WITH_CUDA)
	set(DALI_BINARY_DIR $ENV{DALI_HOME}/build)
else()
	set(DALI_BINARY_DIR $ENV{DALI_HOME}/build_cpu)
	add_definitions(-DMSHADOW_USE_CUDA=0)
endif(WITH_CUDA)

add_subdirectory($ENV{DALI_HOME} ${DALI_BINARY_DIR})

include_directories($ENV{DALI_HOME})
include_directories($ENV{DALI_HOME}/dali)
include_directories(${GFLAGS_INCLUDE_DIRS})
add_definitions(-DMSHADOW_USE_CBLAS)
add_definitions(-DMSHADOW_USE_MKL=0)
INCLUDE_DIRECTORIES($ENV{DALI_HOME}/third_party/mshadow)
include_directories($ENV{DALI_HOME}/third_party/SQLiteCpp/include)
include_directories($ENV{DALI_HOME}/third_party/json11)
include_directories($ENV{DALI_HOME}/third_party/libcuckoo/src)
include_directories($ENV{DALI_HOME}/third_party/sole)
include_directories($ENV{DALI_HOME}/third_party/redox/include)

# ensure we are using C++11
# -stdlib=libc++
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -O3 -g" )













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

foreach(program ${EXAMPLE_PROGRAMS})
    add_executable(${program} ${EXAMPLES_DIR}/${program}.cpp)
    target_link_libraries(${program} dali)
    if (MKL_FOUND AND BLAS_FOUND)
        add_custom_command(TARGET ${program}
            COMMAND sh ${PROJECT_SOURCE_DIR}/scripts/fix_dylib.sh ${EXAMPLES_BUILD_DIR}/${program}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
    endif (MKL_FOUND AND BLAS_FOUND)
endforeach(program)
