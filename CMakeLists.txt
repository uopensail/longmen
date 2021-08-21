cmake_minimum_required(VERSION 3.17)
project(longmen)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(third/luban/include)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_subdirectory(third/luban)

set(SOURCE src/common.cpp src/config.cpp src/loader.cpp
        src/longmen.cpp src/mutils.cpp src/ps.cpp src/rank.cpp)

set(INCLUDE include/common.h include/common.h include/loader.h
        include/longmen.h include/mutils.h include/ps.h include/rank.h)

add_library(longmen SHARED ${INCLUDE} ${SOURCE})

add_executable(test ${INCLUDE} ${SOURCE} src/test.cpp)

add_executable(tf_tool src/tf_tool.cpp)

target_link_libraries(longmen protobuf tcmalloc toolkit zip tensorflow stata glog)
target_link_libraries(test protobuf toolkit zip tensorflow stata glog)
target_link_libraries(tf_tool tensorflow)