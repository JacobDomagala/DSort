cmake_minimum_required(VERSION 3.22)

# Hardcoded CUDA path
set(CUDA_ROOT "/usr/local/cuda-12.6")

# Force CMake to use your specific nvcc
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA Compiler" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_ROOT}" CACHE PATH "CUDA Toolkit Root" FORCE)


project(DSort VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Source file
add_executable(DSort
    sort.cu
)

target_link_libraries(DSort
    PRIVATE cudart curand
)

# Compiler warnings
target_compile_options(DSort
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)

set_target_properties(DSort PROPERTIES
    CUDA_ARCHITECTURES "75;86;89"
)
