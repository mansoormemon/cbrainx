set(MISC "misc")
option(CBRAINX_BUILD_MISC_EXAMPLES "Build miscellaneous examples." OFF)

if(CBRAINX_BUILD_MISC_EXAMPLES)
    add_subdirectory("${MISC}")
endif()

set(DIGIT_RECOGNIZER "digitRecognizer")

add_subdirectory("${DIGIT_RECOGNIZER}")
