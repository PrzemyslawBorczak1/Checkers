cmake_minimum_required(VERSION 3.22)

project(checkers LANGUAGES CXX CUDA)

# Standardy
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation (RDC) - potrzebne jeśli masz device code w wielu plikach
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(checkers
    main.cpp
    common.cpp
    GPU.cpp
    human_player.cpp
    mcts_player.cpp
    mcts_kernel.cu
)

target_include_directories(checkers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Warningi tylko dla CXX (nie dla CUDA/nvcc!)
if (MSVC)
    target_compile_options(checkers PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(checkers PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

# Architektura GPU (możesz zmienić na jedną, zgodną z Twoją kartą)
set_property(TARGET checkers PROPERTY CUDA_ARCHITECTURES 86)

# CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(checkers PRIVATE CUDA::cudart)

# (opcjonalnie) Jeśli nvcc krzyczy o MSVC, możesz wymusić konkretny toolset w komendzie CMake (-T)
