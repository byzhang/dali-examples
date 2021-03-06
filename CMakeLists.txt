cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("dali-examples")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS "${CMAKE_EXPORT_COMPILE_COMMANDS}" CACHE STRING "export compile_commands.json")

# additional cmake find files
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
message("CMAKE_MODULE_PATH " ${CMAKE_MODULE_PATH})
message("CMAKE_PREFIX_PATH " ${CMAKE_PREFIX_PATH})

# find Dali.
if (DEFINED ENV{DALI_HOME} AND NOT "$ENV{DALI_HOME}" STREQUAL "")
    set(Dali_DIR $ENV{DALI_HOME}/cmake)
endif()
find_package(Dali REQUIRED)
include_directories(${DALI_AND_DEPS_INCLUDE_DIRS} $ENV{DALI_HOME})

# find gflags
find_package(Gflags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})
add_definitions(-DGFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})

add_subdirectory(${PROJECT_SOURCE_DIR}/visualizer-clients/cpp)
include_directories(${PROJECT_SOURCE_DIR}/visualizer-clients/cpp)
include_directories(${PROJECT_SOURCE_DIR}/visualizer-clients/cpp/third_party/json11)
include_directories(${PROJECT_SOURCE_DIR}/visualizer-clients/cpp/third_party/redox/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/SQLiteCpp)
include_directories(${PROJECT_SOURCE_DIR}/third_party/SQLiteCpp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/SQLiteCpp/sqlite3)

# ensure we are using C++11
# TODO(jonathan): is this flag required for apple?
# -stdlib=libc++
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -O3 -g -fPIC" )


# compile protobufs defined by us.
add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf)
include_directories(${PROJECT_SOURCE_DIR})

# compile utils for examples - handling protobufs and sqlite etc.
file(GLOB_RECURSE UTILS_FILES "${PROJECT_SOURCE_DIR}/models/*.cpp")
list(APPEND       UTILS_FILES "examples/utils.h")
add_library(utils examples/utils.cpp ${UTILS_FILES})
target_link_libraries(utils SQLiteCpp sqlite3 proto ${CMAKE_DL_LIBS})

# Compile all the examples
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
    target_link_libraries(${program} ${DALI_AND_DEPS_LIBRARIES} ${GFLAGS_LIBRARIES}
      dali_visualizer
      utils
      z
    )

    # TODO(jonathan): fix this. preferably move to dali, or at least make
    # a cmake function provided by find dali
    # if (MKL_FOUND AND BLAS_FOUND)
    #     add_custom_command(TARGET ${program}
    #         COMMAND sh ${PROJECT_SOURCE_DIR}/scripts/fix_dylib.sh ${EXAMPLES_BUILD_DIR}/${program}
    #         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
    # endif (MKL_FOUND AND BLAS_FOUND)
endforeach(program)
