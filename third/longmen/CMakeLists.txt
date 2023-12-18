cmake_minimum_required(VERSION 3.16)
project(longmen)

set(CMAKE_CXX_STANDARD 17)

include_directories(../sample-luban/include)
include_directories(../sample-luban/luban/include)
include_directories(/usr/local/include)
include_directories(/usr/include)
include_directories(include)
link_directories(/usr/local/lib)

# Find Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

 
add_subdirectory(../sample-luban ./sample-luban)

SET(LONGMEN_SOURCE src/longmen.cpp src/model.cpp )

add_library(longmen_static STATIC ${LONGMEN_SOURCE})
target_link_libraries(longmen_static luban_static sample_luban_static "${TORCH_LIBRARIES}")
set_property(TARGET longmen_static PROPERTY CXX_STANDARD 17)

add_library(longmen SHARED ${LONGMEN_SOURCE})
target_link_libraries(longmen c10 torch_cpu luban_static sample_luban_static)



