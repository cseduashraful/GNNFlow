cmake_minimum_required(VERSION 3.18)
project(libgnnflow CXX)
enable_language(CUDA)

set(TARGET_LIB "libgnnflow")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# set cuda arch
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
find_package(CUDAToolkit REQUIRED)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS)
if (NOT INSTALLED_GPU_CCS)
    set(INSTALLED_GPU_CCS "75")
endif()
string(STRIP "${INSTALLED_GPU_CCS}" INSTALLED_GPU_CCS)
string(REPLACE " " ";" INSTALLED_GPU_CCS "${INSTALLED_GPU_CCS}")
set(CMAKE_CUDA_ARCHITECTURES "")
foreach(arch IN LISTS INSTALLED_GPU_CCS)
    string(REPLACE "." "" arch "${arch}")
    string(REPLACE "+PTX" "" arch "${arch}")
    list(APPEND CMAKE_CUDA_ARCHITECTURES "${arch}")
endforeach()
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)
message(STATUS "CUDA_ARCH_LIST: ${CMAKE_CUDA_ARCHITECTURES}")

# 3rd party
set(PYBIND11_FINDPYTHON ON)
if (DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(third_party/pybind11)


include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/gnnflow/csrc)
include_directories(third_party/pybind11/include)
include_directories(third_party/spdlog/include)
include_directories(third_party/rmm/include)


file(GLOB_RECURSE GNNFLOW_SRC_FILES ${PROJECT_SOURCE_DIR}/gnnflow/csrc/*.cc)
set_source_files_properties(${PROJECT_SOURCE_DIR}/gnnflow/csrc/api.cc PROPERTIES LANGUAGE CUDA)
file(GLOB_RECURSE GNNFLOW_SRC_CUDA_FILES ${PROJECT_SOURCE_DIR}/gnnflow/csrc/*.cu)
set_source_files_properties(${GNNFLOW_SRC_CUDA_FILES} PROPERTIES LANGUAGE CUDA)
list(APPEND GNNFLOW_SRC_FILES ${GNNFLOW_SRC_CUDA_FILES})

pybind11_add_module(${TARGET_LIB} ${GNNFLOW_SRC_FILES})

add_subdirectory(third_party/abseil-cpp)
target_link_libraries(${TARGET_LIB} PRIVATE absl::flat_hash_map)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(${TARGET_LIB} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} CUDA::curand)

target_compile_options(${TARGET_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    -rdc=true
    -fopenmp
    >)

set_property(TARGET ${TARGET_LIB} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp -fPIC -Wall -ftree-vectorize")
set(ARCH_FLAGS "-march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
