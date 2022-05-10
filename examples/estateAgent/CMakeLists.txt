cmake_minimum_required(VERSION 3.16)

project(estateAgent)

set(RESOURCES "res")

file(COPY "${RESOURCES}"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

set(TARGET "estateAgent")
set(SOURCES "src/main.cc")
set(INCLUDES "src/rapidcsv/rapidcsv.h")

set(LIBFMT "fmt")
set(LIBFMT_INCLUDE_DIR "${CBRAINX_EXTERNAL_DIR}/${LIBFMT}/include")

add_executable("${TARGET}" "${INCLUDES}" "${SOURCES}" "${INCLUDES}")

target_include_directories("${TARGET}" PUBLIC "${CBRAINX_INCLUDE_DIR}")

target_link_libraries("${TARGET}" "${CBRAINX}" "${LIBFMT}")
