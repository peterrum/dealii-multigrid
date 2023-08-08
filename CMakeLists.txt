CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT("examples")

ADD_LIBRARY(examples dummy.cc)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
)

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
)

# Set available operators
set(VARIANTS_OP "LAPLACE;ELASTICITY")

set(OPTION_OP_LAPLACE "OPERATOR=0")
set(OPTION_OP_ELASTICITY "OPERATOR=1")

foreach(VAR_OP ${VARIANTS_OP})
  set(executable_name "multigrid_throughput-${VAR_OP}")
  add_executable(${executable_name} "multigrid_throughput.cc")
  DEAL_II_SETUP_TARGET(${executable_name})
  target_compile_definitions(${executable_name} PUBLIC
    -D${OPTION_OP_${VAR_OP}})
endforeach()

target_link_libraries(${executable_name} "examples")
