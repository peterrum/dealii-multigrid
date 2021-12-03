SET(TARGET "main")

SET(TARGET_SRC main.cc)

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
PROJECT("exmples")

DEAL_II_INITIALIZE_CACHED_VARIABLES()

file(GLOB files "*.cc")
foreach(file ${files})
  get_filename_component(app ${file} NAME_WLE)

  ADD_EXECUTABLE(${app} ${file})
  TARGET_COMPILE_DEFINITIONS(${app} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  DEAL_II_SETUP_TARGET(${app})

endforeach()

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

enable_testing()

DEAL_II_PICKUP_TESTS("tests")


