set(MISC "misc")
option(CBRAINX_BUILD_MISC_EXAMPLES "Build miscellaneous examples." OFF)

if(CBRAINX_BUILD_MISC_EXAMPLES)
    add_subdirectory("${MISC}")
endif()

set(DIGIT_RECOGNIZER "digitRecognizer")

add_subdirectory("${DIGIT_RECOGNIZER}")

set(ESTATE_AGENT "estateAgent")

add_subdirectory("${ESTATE_AGENT}")

set(IONOSPHERE "ionosphere")

add_subdirectory("${IONOSPHERE}")

set(PIMA_INDIANS_DIABETES "pimaIndiansDiabetes")

add_subdirectory("${PIMA_INDIANS_DIABETES}")
