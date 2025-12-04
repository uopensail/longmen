# ==============================================================================
# LongMen - ONNX Model Inference Library
# ==============================================================================

cmake_minimum_required(VERSION 3.16)

project(longmen
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "High-performance ONNX inference library with feature processing"
)

# ==============================================================================
# Build Options
# ==============================================================================
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(ENABLE_GLOG "Enable glog logging" ON)

# ==============================================================================
# C++ Standard
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ==============================================================================
# Compiler Flags
# ==============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Common flags
    add_compile_options(-fPIC -Wall -Wextra)

    # Debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

    # Release flags with optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# ==============================================================================
# Platform Detection
# ==============================================================================
if(APPLE)
    set(PLATFORM_NAME "macOS")
    set(SHARED_LIB_EXT ".dylib")
    add_compile_definitions(_LIBCPP_DISABLE_AVAILABILITY)
    set(CMAKE_MACOSX_RPATH ON)
elseif(UNIX)
    set(PLATFORM_NAME "Linux")
    set(SHARED_LIB_EXT ".so")
elseif(WIN32)
    set(PLATFORM_NAME "Windows")
    set(SHARED_LIB_EXT ".dll")
    add_compile_definitions(_WIN32_WINNT=0x0601)
endif()

message(STATUS "Platform: ${PLATFORM_NAME}")

# ==============================================================================
# Installation Paths
# ==============================================================================
include(GNUInstallDirs)

# Set default installation prefix to /usr/local if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix" FORCE)
endif()

# Define installation directories
set(LONGMEN_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
set(LONGMEN_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include/longmen")
set(LONGMEN_INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}/bin")
set(LONGMEN_INSTALL_CMAKEDIR "${CMAKE_INSTALL_PREFIX}/lib/cmake/longmen")

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# ==============================================================================
# ONNX Runtime Configuration
# ==============================================================================

# Set ONNX Runtime paths (customize if needed)
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime root directory")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Determine library name by platform
if(APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib")
elseif(UNIX)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
else()
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
endif()

# Verify library exists
if(NOT EXISTS ${ONNXRUNTIME_LIB})
    message(FATAL_ERROR "ONNX Runtime not found at: ${ONNXRUNTIME_LIB}")
endif()

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

# ==============================================================================
# Third-Party Dependencies
# ==============================================================================
message(STATUS "Configuring dependencies...")

# Include dependency scripts
include(minia/cmake/antlr4.cmake)
include(minia/cmake/gflags.cmake)

# Configure dependencies
configure_antlr4_runtime()
configure_gflags()

# Configure glog if enabled
if(ENABLE_GLOG)
    include(minia/cmake/glog.cmake)
    configure_glog()
    message(STATUS "glog logging: ENABLED")
else()
    message(STATUS "glog logging: DISABLED")
endif()

# System dependencies
find_package(Threads REQUIRED)

message(STATUS "Dependencies configured")

# ==============================================================================
# Source Files
# ==============================================================================

# Minia core sources
set(MINIA_SOURCES
    minia/src/common.cc
    minia/src/builtin.cc
    minia/src/minia.cc
    minia/src/MurmurHash3.cc
)

# Minia grammar-generated sources
set(MINIA_GRAMMAR_SOURCES
    minia/src/grammar/minia.cc
    minia/src/grammar/miniaBaseListener.cpp
    minia/src/grammar/miniaLexer.cpp
    minia/src/grammar/miniaListener.cpp
    minia/src/grammar/miniaParser.cpp
)

# LongMen core sources
set(LONGMEN_SOURCES
    longmen/src/arena.cc
    longmen/src/graph.cc
    longmen/src/longmen.cc
    longmen/src/model.cc
    longmen/src/placement.cc
    longmen/src/onnx_ops.cc
)

# Combine all sources
set(ALL_SOURCES
    ${MINIA_SOURCES}
    ${MINIA_GRAMMAR_SOURCES}
    ${LONGMEN_SOURCES}
)

# ==============================================================================
# Common Target Configuration
# ==============================================================================
function(configure_target target_name)
    # Include directories
    target_include_directories(${target_name}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/longmen/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minia/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/minia/include/grammar>
        $<BUILD_INTERFACE:${ONNXRUNTIME_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include/longmen>
        PRIVATE
        ${ANTLR4_INCLUDE_DIR}
        ${GFLAGS_INCLUDE_DIR}
        $<$<BOOL:${ENABLE_GLOG}>:${GLOG_INCLUDE_DIR}>
    )

    # Link directories
    target_link_directories(${target_name} PRIVATE
        ${ONNXRUNTIME_LIB_DIR}
        ${ANTLR4_LIBRARY_DIR}
        ${GFLAGS_LIBRARY_DIR}
        $<$<BOOL:${ENABLE_GLOG}>:${GLOG_LIBRARY_DIR}>
    )

    # Prepare link libraries list
    set(LINK_LIBS
        ${ONNXRUNTIME_LIB}
        ${ANTLR4_LIBRARY_DIR}/libantlr4-runtime.a
        ${GFLAGS_LIBRARY_DIR}/libgflags.a
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )

    # Add glog if enabled
    if(ENABLE_GLOG)
        list(APPEND LINK_LIBS ${GLOG_LIBRARY_DIR}/libglog.a)
        target_compile_definitions(${target_name} PRIVATE ENABLE_GLOG)
    endif()

    # Link libraries
    target_link_libraries(${target_name} PRIVATE ${LINK_LIBS})

    # Common properties
    set_target_properties(${target_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET default
    )
endfunction()

# ==============================================================================
# Shared Library
# ==============================================================================
if(BUILD_SHARED_LIBS)
    message(STATUS "Configuring shared library...")

    add_library(longmen_shared SHARED ${ALL_SOURCES})
    configure_target(longmen_shared)

    # Shared library properties
    set_target_properties(longmen_shared PROPERTIES
        OUTPUT_NAME longmen
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    # RPATH configuration for runtime library search
    if(UNIX)
        if(APPLE)
            set_target_properties(longmen_shared PROPERTIES
                BUILD_WITH_INSTALL_RPATH OFF
                BUILD_RPATH_USE_ORIGIN ON
                INSTALL_RPATH "@loader_path;/usr/local/lib;${ONNXRUNTIME_LIB_DIR}"
            )
        else()
            set_target_properties(longmen_shared PROPERTIES
                BUILD_WITH_INSTALL_RPATH OFF
                BUILD_RPATH_USE_ORIGIN ON
                INSTALL_RPATH "$ORIGIN:/usr/local/lib:${ONNXRUNTIME_LIB_DIR}"
            )
        endif()
    endif()

    list(APPEND INSTALL_TARGETS longmen_shared)
    message(STATUS "Shared library configured")
endif()

# ==============================================================================
# Installation
# ==============================================================================
if(INSTALL_TARGETS)
    message(STATUS "Configuring installation...")

    # Install library targets
    install(TARGETS ${INSTALL_TARGETS}
        EXPORT longmen-targets
        LIBRARY DESTINATION ${LONGMEN_INSTALL_LIBDIR}
        COMPONENT Runtime
        NAMELINK_COMPONENT Development
        ARCHIVE DESTINATION ${LONGMEN_INSTALL_LIBDIR}
        COMPONENT Development
        RUNTIME DESTINATION ${LONGMEN_INSTALL_BINDIR}
        COMPONENT Runtime
    )

    # Install public headers
    install(DIRECTORY longmen/include/
        DESTINATION ${LONGMEN_INSTALL_INCLUDEDIR}
        COMPONENT Development
        FILES_MATCHING PATTERN "*.h"
        PATTERN "*.hpp"
    )

    # Install minia headers (if needed by public API)
    install(DIRECTORY minia/include/
        DESTINATION ${LONGMEN_INSTALL_INCLUDEDIR}/minia
        COMPONENT Development
        FILES_MATCHING PATTERN "*.h"
        PATTERN "*.hpp"
    )

    # Install ONNX Runtime library (if shared build)
    if(BUILD_SHARED_LIBS AND EXISTS ${ONNXRUNTIME_LIB})
        install(FILES ${ONNXRUNTIME_LIB}
            DESTINATION ${LONGMEN_INSTALL_LIBDIR}
            COMPONENT Runtime
        )
        message(STATUS "Will install ONNX Runtime library")
    endif()

    # Create and install CMake package config files
    include(CMakePackageConfigHelpers)

    # Generate version file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/longmen-config-version.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Generate config file
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/longmen-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/longmen-config.cmake"
        INSTALL_DESTINATION ${LONGMEN_INSTALL_CMAKEDIR}
        PATH_VARS LONGMEN_INSTALL_INCLUDEDIR LONGMEN_INSTALL_LIBDIR
    )

    # Install CMake config files
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/longmen-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/longmen-config-version.cmake"
        DESTINATION ${LONGMEN_INSTALL_CMAKEDIR}
        COMPONENT Development
    )

    # Install export targets
    install(EXPORT longmen-targets
        FILE longmen-targets.cmake
        NAMESPACE longmen::
        DESTINATION ${LONGMEN_INSTALL_CMAKEDIR}
        COMPONENT Development
    )

    # Create pkg-config file
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/longmen.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/longmen.pc"
        @ONLY
    )

    # Install pkg-config file
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/longmen.pc"
        DESTINATION ${LONGMEN_INSTALL_LIBDIR}/pkgconfig
        COMPONENT Development
    )

    message(STATUS "Installation configured:")
    message(STATUS "  Libraries:  ${LONGMEN_INSTALL_LIBDIR}")
    message(STATUS "  Headers:    ${LONGMEN_INSTALL_INCLUDEDIR}")
    message(STATUS "  CMake:      ${LONGMEN_INSTALL_CMAKEDIR}")
endif()

# ==============================================================================
# Uninstall Target
# ==============================================================================

# Create uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
        COMMENT "Uninstalling LongMen..."
    )
endif()

# ==============================================================================
# Build Summary
# ==============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "LongMen Build Configuration")
message(STATUS "========================================")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "Platform:         ${PLATFORM_NAME}")
message(STATUS "Compiler:         ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "  glog logging:   ${ENABLE_GLOG}")
message(STATUS "")
message(STATUS "Installation:")
message(STATUS "  Prefix:         ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Libraries:      ${LONGMEN_INSTALL_LIBDIR}")
message(STATUS "  Headers:        ${LONGMEN_INSTALL_INCLUDEDIR}")
message(STATUS "  CMake config:   ${LONGMEN_INSTALL_CMAKEDIR}")
message(STATUS "========================================")
message(STATUS "")
