set(MISC "misc")
option(CBRAINX_BUILD_MISC_EXAMPLES "Build miscellaneous examples." ON)

if(CBRAINX_BUILD_MISC_EXAMPLES)
    add_subdirectory("${MISC}")
endif()
