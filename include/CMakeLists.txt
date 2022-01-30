set(CBRAINX_INCLUDES
    "cbrainx/cbrainx.hh"
    "cbrainx/shape.hh"
    "cbrainx/type_aliases.hh"
    "cbrainx/version.hh")

if(CBRAINX_INSTALL)
    set(CBRAINX_INSTALL_INCLUDEDIR "include")

    install(DIRECTORY "${CBRAINX_DIR}"
            DESTINATION "${CBRAINX_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.hh")
endif()
