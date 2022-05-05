set(CBRAINX_INCLUDES
    "cbrainx/abstractLayer.hh"
    "cbrainx/activationFunctions.hh"
    "cbrainx/activationLayer.hh"
    "cbrainx/cbrainx.hh"
    "cbrainx/customViews.hh"
    "cbrainx/denseLayer.hh"
    "cbrainx/exceptions.hh"
    "cbrainx/image.hh"
    "cbrainx/imgProc.hh"
    "cbrainx/iterators.hh"
    "cbrainx/lossFunctions.hh"
    "cbrainx/neuralNet.hh"
    "cbrainx/optimizers.hh"
    "cbrainx/shape.hh"
    "cbrainx/softmax.hh"
    "cbrainx/stopwatch.hh"
    "cbrainx/tensor.hh"
    "cbrainx/typeAliases.hh"
    "cbrainx/typeConcepts.hh"
    "cbrainx/version.hh")

if(CBRAINX_INSTALL)
    set(CBRAINX_INSTALL_INCLUDEDIR "include")

    install(DIRECTORY "${CBRAINX_DIR}"
            DESTINATION "${CBRAINX_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.hh")
endif()
