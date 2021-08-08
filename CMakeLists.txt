cmake_minimum_required(VERSION 3.17)
project(longmen)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(third/luban/include)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
add_library(longmen include/common.h include/loader.h include/config.h src/rank.cpp include/rank.h include/ps.h src/common.cpp src/ps.cpp src/loader.cpp src/mutils.cpp include/mutils.h src/config.cpp)