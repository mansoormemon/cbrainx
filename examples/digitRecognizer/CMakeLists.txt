cmake_minimum_required(VERSION 3.16)

project(digitRecognizer)

set(RESOURCES "res")

file(COPY "${RESOURCES}"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

set(TARGET "digitRecognizer")
set(SOURCES "src/main.cc")

set(LIBFMT "fmt")
set(LIBFMT_INCLUDE_DIR "${CBRAINX_EXTERNAL_DIR}/${LIBFMT}/include")

add_executable("${TARGET}" "${INCLUDES}" "${SOURCES}")

target_include_directories("${TARGET}" PUBLIC "${CBRAINX_INCLUDE_DIR}")

target_link_libraries("${TARGET}" "${CBRAINX}" "${LIBFMT}")
