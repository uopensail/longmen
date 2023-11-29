cmake_minimum_required(VERSION 3.16)
project(longmen)

set(CMAKE_CXX_STANDARD 17)

include_directories(../luban/include)
include_directories(/usr/local/include)
include_directories(/usr/include)
include_directories(include)
link_directories(/usr/local/lib)
link_directories(/usr/local/lib/libtorch/lib)

# Set the path to the PyTorch installation directory
set(Torch_DIR /usr/local/lib/libtorch/share/cmake/Torch)

# Find Torch
find_package(Torch REQUIRED)

SET(LUBAN_SOURCE  ../luban/src/builtin.cpp ../luban/src/common.cpp 
../luban/src/feature.cpp ../luban/src/MurmurHash3.cc 
../luban/src/operator.cpp ../luban/src/placement.cpp 
../luban/src/toolkit.cpp)

SET(LONGMEN_SOURCE src/longmen.cpp src/model.cpp ${LUBAN_SOURCE})

add_library(longmen SHARED ${LONGMEN_SOURCE})
target_link_libraries(longmen c10 torch_cpu)

add_library(longmen_static STATIC ${LONGMEN_SOURCE})
target_link_libraries(longmen_static c10 torch_cpu longmen)


