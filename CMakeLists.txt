# Disable in-source builds to prevent source tree corruption.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       Create a separate directory for build files.")
endif()

cmake_minimum_required(VERSION 3.16)

if("${CMAKE_VERSION}" VERSION_LESS 3.21)
    cmake_policy(VERSION "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}")
else()
    cmake_policy(VERSION 3.21)
endif()

if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    set(CBRAINX_IS_MASTER_PROJECT ON)
else()
    set(CBRAINX_IS_MASTER_PROJECT OFF)
endif()

# Set the build type to "Release" if none is specified explicitly.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

project(cbrainx
        VERSION 0.1.0
        DESCRIPTION "A C++ library for working with neural networks."
        HOMEPAGE_URL "https://github.com/mansoormemon/cbrainx"
        LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options("-Wall" "-Wextra")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options("/W4")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CBRAINX "cbrainx" CACHE INTERNAL "CBrainX")

set(CBRAINX_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" CACHE INTERNAL
    "Path to CBrainX includes directory.")

set(CBRAINX_DIR "cbrainx")

set(CBRAINX_RESOURCES_DIR "${PROJECT_SOURCE_DIR}/res")

option(CBRAINX_INSTALL "Generate CBrainX install targets." "${CBRAINX_IS_MASTER_PROJECT}")

set(CBRAINX_EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
set(CBRAINX_EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external")
set(CBRAINX_SOURCES_DIR "${PROJECT_SOURCE_DIR}/src")

add_subdirectory("${CBRAINX_INCLUDE_DIR}")
add_subdirectory("${CBRAINX_SOURCES_DIR}")
add_subdirectory("${CBRAINX_EXTERNAL_DIR}")

option(CBRAINX_BUILD_EXAMPLES "Build CBrainX examples." "${CBRAINX_IS_MASTER_PROJECT}")

if(CBRAINX_BUILD_EXAMPLES)
    add_subdirectory("${CBRAINX_EXAMPLES_DIR}")
endif()

message("
-----------------------------------
CBRAINX BUILD SUMMARY:
-----------------------------------
PROJECT                     : ${PROJECT_NAME} (v${PROJECT_VERSION})
PROJECT_PATH                : ${PROJECT_SOURCE_DIR}
LANGUAGE(S)                 : C++${CMAKE_CXX_STANDARD}
PLATFORM                    : ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_VERSION} - ${CMAKE_SYSTEM_PROCESSOR})
BUILD_TYPE                  : ${CMAKE_BUILD_TYPE}
INSTALL_PREFIX              : ${CMAKE_INSTALL_PREFIX}
COMPILER                    : ${CMAKE_CXX_COMPILER_ID} => ${CMAKE_CXX_COMPILER} (v${CMAKE_CXX_COMPILER_VERSION})
MAKE_PROGRAM                : ${CMAKE_MAKE_PROGRAM}
---
GENERATE_INSTALL_TARGETS    : ${CBRAINX_INSTALL}
BUILD_EXAMPLES              : ${CBRAINX_BUILD_EXAMPLES}
-----------------------------------
")
