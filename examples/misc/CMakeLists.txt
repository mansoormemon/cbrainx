set(EXAMPLES
    "bmMatrix.cc"
    "exActFuncs.cc"
    "exImage.cc"
    "exImgProc.cc"
    "exLossFunctions.cc"
    "exNet.cc"
    "exShape.cc"
    "exSoftmax.cc"
    "exTensor.cc"
    "exTensorFun.cc")

set(LIBFMT "fmt")
set(LIBFMT_INCLUDE_DIR "${CBRAINX_EXTERNAL_DIR}/${LIBFMT}/include")

foreach(EG_FILE ${EXAMPLES})
    get_filename_component(EXAMPLE "${EG_FILE}" NAME_WE)
    add_executable("${EXAMPLE}" "${EG_FILE}")

    target_include_directories("${EXAMPLE}"
                               PUBLIC
                               "${CBRAINX_INCLUDE_DIR}"
                               "${LIBFMT_INCLUDE_DIR}")

    target_link_libraries("${EXAMPLE}" PUBLIC "${CBRAINX}" "${LIBFMT}")
endforeach()
