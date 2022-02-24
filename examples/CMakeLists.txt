set(EXAMPLES
    "bm_matrix.cc"
    "ex_act_funcs.cc"
    "ex_image.cc"
    "ex_img_proc.cc"
    "ex_matrix.cc"
    "ex_matrix_fun.cc"
    "ex_net.cc"
    "ex_shape.cc"
    "ex_soft_max.cc"
    "ex_table.cc"
    "ex_tensor.cc"
    "ex_tensor_fun.cc")

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
