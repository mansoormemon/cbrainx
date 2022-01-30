set(EXAMPLES
    "ex_shape.cc")

foreach(EG_FILE ${EXAMPLES})
    get_filename_component(EXAMPLE "${EG_FILE}" NAME_WE)
    add_executable("${EXAMPLE}" "${EG_FILE}")

    target_include_directories("${EXAMPLE}"
                               PUBLIC
                               "${CBRAINX_INCLUDE_DIR}")

    target_link_libraries("${EXAMPLE}" PUBLIC "${CBRAINX}")
endforeach()
