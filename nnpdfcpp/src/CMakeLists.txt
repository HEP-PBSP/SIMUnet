# common files
add_library(common STATIC
  common/src/nnpdfsettings.cc
  common/src/md5.cc
  common/src/loadutils.cc
  common/src/datautils.cc
)
include_directories(common/inc)

# filter
option(COMPILE_filter "Compile filter program." ON)
if (COMPILE_filter)
  add_executable(filter filter/src/filter.cc
                      filter/src/kincuts.cc)
  include_directories(filter/inc)
  target_link_libraries(filter common nnpdf ${LHAPDF_LIBRARIES} ${YAML_LDFLAGS} ${GSL_LDFLAGS})
  install(TARGETS filter DESTINATION bin
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif(COMPILE_filter)

# nnfit
option(COMPILE_nnfit "Compile nnfit program." ON)
if (COMPILE_nnfit)
  add_executable(nnfit nnfit/src/nnfit.cc
                       nnfit/src/apfelevol.cc
                       nnfit/src/fastaddchi2.cc
                       nnfit/src/fitbases.cc
                       nnfit/src/fitpdfset.cc
                       nnfit/src/minimizer.cc
                       nnfit/src/pdfbasis.cc
                       nnfit/src/stopping.cc)
  include_directories(nnfit/inc)
  target_link_libraries(nnfit common nnpdf ${LHAPDF_LIBRARIES} ${YAML_LDFLAGS} ${APFEL_LIBRARIES} ${GSL_LDFLAGS})
  install(TARGETS nnfit DESTINATION bin
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif(COMPILE_nnfit)

# chi2check
option(COMPILE_chi2check "Compile chi2check program." ON)
if (COMPILE_chi2check)
  add_executable(chi2check chi2check/src/chi2check.cc)
  target_link_libraries(chi2check common nnpdf ${LHAPDF_LIBRARIES} ${YAML_LDFLAGS} ${GSL_LDFLAGS})
  install(TARGETS chi2check DESTINATION bin
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif(COMPILE_chi2check)

# ROOT
#find_program(ROOT_CONFIG root-config)
#if (ROOT_CONFIG)
#  exec_program(${ROOT_CONFIG}
#    ARGS --cflags
#    OUTPUT_VARIABLE ROOT_CXX_FLAGS
#    )
#  exec_program(${ROOT_CONFIG}
#    ARGS --libs
#    OUTPUT_VARIABLE ROOT_LIBRARIES
#    )
#  # validphys
#  option(COMPILE_validphys "Compile validphys program." OFF)
#  if (COMPILE_validphys)  
#    add_executable(validphys validphys/src/validphys.cc
#      validphys/src/plotdata.cc
#      validphys/src/plotutils.cc)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROOT_CXX_FLAGS}")
#    include_directories(validphys/inc)
#    target_link_libraries(validphys common nnpdf ${LHAPDF_LIBRARIES} ${ROOT_LIBRARIES} ${YAML_LDFLAGS} ${GSL_LDFLAGS})
#    install(TARGETS validphys DESTINATION bin
#          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
#  endif(COMPILE_validphys)
#else(ROOT_CONFIG)
#  message(WARNING "ROOT not found: validphys compilation is disabled, install ROOT and rerun cmake to enable the validphys compilation")
#endif(ROOT_CONFIG)

# revolve
option(COMPILE_revolve "Compile revolve program." OFF)
if (COMPILE_revolve)
  add_executable(revolve revolve/src/revolve.cc)
  target_link_libraries(revolve common nnpdf ${LHAPDF_LIBRARIES} ${YAML_LDFLAGS} ${ROOT_LIBRARIES} ${APFEL_LIBRARIES} ${GSL_LDFLAGS})
  install(TARGETS revolve DESTINATION bin
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)  
endif(COMPILE_revolve)

# revolvenet
option(COMPILE_revolvenet "Compile revolvenet program." OFF)
if (COMPILE_revolvenet)
  add_executable(revolvenet revolvenet/src/revolvenet.cc
    nnfit/src/apfelevol.cc revolvenet/src/nnpdf.cc
    nnfit/src/fitbases.cc)
  include_directories(revolvenet/inc nnfit/inc)
  target_link_libraries(revolvenet common nnpdf ${LHAPDF_LIBRARIES} ${YAML_LDFLAGS} ${ROOT_LIBRARIES} ${APFEL_LIBRARIES} ${GSL_LDFLAGS})
  install(TARGETS revolvenet DESTINATION bin
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)  
endif(COMPILE_revolvenet)

# libfiatlux
find_program(FIATLUX_CONFIG fiatlux-config)
if (FIATLUX_CONFIG)
  exec_program(${FIATLUX_CONFIG}
    ARGS --cppflags
    OUTPUT_VARIABLE FIATLUX_CXX_FLAGS
    )
  exec_program(${FIATLUX_CONFIG}
    ARGS --ldflags
    OUTPUT_VARIABLE FIATLUX_LIBRARIES
    )
  # fiatlux
  option(COMPILE_fiatlux "Compile fiatlux program." OFF)
  if (COMPILE_fiatlux)
    add_executable(fiatlux fiatlux/src/fiatlux.cc)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIATLUX_CXX_FLAGS}")
    target_link_libraries(fiatlux common nnpdf ${LHAPDF_LIBRARIES} ${YAML_LDFLAGS} ${GSL_LDFLAGS} ${FIATLUX_LIBRARIES} ${APFEL_LIBRARIES})
    install(TARGETS fiatlux DESTINATION bin
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif(COMPILE_fiatlux)
else(FIATLUX_CONFIG)
  message(WARNING "libfiatlux not found: fiatlux compilation is disabled, install libfiatlux and rerun cmake to enable the fiatlux compilation")  
endif(FIATLUX_CONFIG)