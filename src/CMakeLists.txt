option(BUILD_SHARED_LIBS "Build shared/dynamic library." OFF)

set(CBRAINX_ALIAS "cbrainx::cbrainx")

set(CBRAINX_SOURCES
    "abstract_layer.cc"
    "activation_functions.cc"
    "activation_layer.cc"
    "dataset.cc"
    "dense_layer.cc"
    "exceptions.cc"
    "image.cc"
    "img_proc.cc"
    "loss_functions.cc"
    "matrix.cc"
    "neural_network.cc"
    "shape.cc"
    "soft_max.cc"
    "stopwatch.cc"
    "table.cc")

add_library("${CBRAINX}" "${CBRAINX_SOURCES}")
add_library("${CBRAINX_ALIAS}" ALIAS "${CBRAINX}")

set_target_properties("${CBRAINX}" PROPERTIES
                      VERSION "${PROJECT_VERSION}"
                      SOVERSION "${PROJECT_VERSION_MAJOR}")

set(LIBFMT "fmt")
set(LIBFMT_INCLUDE_DIR "${CBRAINX_EXTERNAL_DIR}/${LIBFMT}/include")

set(LIBSTB "stb")
set(LIBSTB_INCLUDE_DIR "${CBRAINX_EXTERNAL_DIR}/${LIBSTB}")

if(UNIX)
    set(PTHREAD_SWITCH "-pthread")
    target_link_options("${CBRAINX}" PUBLIC "${PTHREAD_SWITCH}")
endif()

target_include_directories("${CBRAINX}"
                           PUBLIC
                           "${CBRAINX_INCLUDE_DIR}"
                           "${LIBFMT_INCLUDE_DIR}"
                           "${LIBSTB_INCLUDE_DIR}")

target_link_libraries("${CBRAINX}" PRIVATE "${LIBFMT}")

if(CBRAINX_INSTALL)
    set(CBRAINX_INSTALL_LIBDIR "lib/${CBRAINX_DIR}")

    install(TARGETS "${CBRAINX}"
            ARCHIVE DESTINATION "${CBRAINX_INSTALL_LIBDIR}"
            LIBRARY DESTINATION "${CBRAINX_INSTALL_LIBDIR}")
endif()
