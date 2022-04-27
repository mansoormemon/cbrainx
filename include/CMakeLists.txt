set(CBRAINX_INCLUDES
    "cbrainx/abstract_layer.hh"
    "cbrainx/activation_functions.hh"
    "cbrainx/activation_layer.hh"
    "cbrainx/cbrainx.hh"
    "cbrainx/custom_iterators.hh"
    "cbrainx/custom_views.hh"
    "cbrainx/dataset.hh"
    "cbrainx/dense_layer.hh"
    "cbrainx/exceptions.hh"
    "cbrainx/image.hh"
    "cbrainx/img_proc.hh"
    "cbrainx/neural_net.hh"
    "cbrainx/shape.hh"
    "cbrainx/softmax.hh"
    "cbrainx/stopwatch.hh"
    "cbrainx/tensor.hh"
    "cbrainx/type_aliases.hh"
    "cbrainx/type_concepts.hh"
    "cbrainx/version.hh")

if(CBRAINX_INSTALL)
    set(CBRAINX_INSTALL_INCLUDEDIR "include")

    install(DIRECTORY "${CBRAINX_DIR}"
            DESTINATION "${CBRAINX_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.hh")
endif()
