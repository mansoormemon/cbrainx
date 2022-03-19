cmake_minimum_required(VERSION 3.16)

project(cbx_ionosphere)

set(IONOSPHERE_RESOURCES "res")

file(COPY "${IONOSPHERE_RESOURCES}"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

set(TARGET "ionosphere")
set(INCLUDES "src/rapidcsv/rapidcsv.h")
set(SOURCES "src/main.cc")

add_executable("${TARGET}" "${INCLUDES}" "${SOURCES}")

target_include_directories("${TARGET}" PUBLIC "${CBRAINX_INCLUDE_DIR}")

target_link_libraries("${TARGET}" "${CBRAINX}")
