cmake_minimum_required(VERSION 3.16)
project(longmen)

set(CMAKE_CXX_STANDARD 17)

include_directories(../third/luban/include)
include_directories(/usr/local/include)
include_directories(/usr/include)

link_directories(/usr/local/lib)

SET(LUBAN_INCLUDE ../third/luban/include/base64.h ../third/luban/include/cpptoml.h ../third/luban/include/builtin_operators.hpp
../third/luban/include/feature_helper.hpp  ../third/luban/include/hash_toolkit.hpp ../third/luban/include/operator_toolkit.hpp
../third/luban/include/runtime.hpp ../third/luban/include/feature.pb.h ../third/luban/include/simple_handler.hpp
../third/luban/include/helper.h ../third/luban/include/MurmurHash3.h ../third/luban/include/toolkit.hpp)

SET(LUBAN_SOURCE  ../third/luban/src/MurmurHash3.cc ../third/luban/src/base64.cpp  ../third/luban/src/feature.pb.cc)

SET(LONGMEN_SOURCE longmen.h longmen.cpp model.hpp ${LUBAN_INCLUDE} ${LUBAN_SOURCE})

add_library(longmen SHARED ${LONGMEN_SOURCE})

target_link_libraries(longmen protobuf pthread c10 torch_cpu)