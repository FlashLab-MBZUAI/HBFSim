cmake_minimum_required(VERSION 3.20)
project(HBFSim VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(
  HBFSIM_EXTERNAL_EVIDENCE_DIR
  ""
  CACHE PATH
  "Digest-bound actual Ramulator2/MQSim evidence directory for L3 validation"
)
set(HBFSIM_EXTERNAL_REPORT_EVIDENCE_ARGS)
set(HBFSIM_CERTIFICATE_EXTERNAL_EVIDENCE_ARGS)
if(HBFSIM_EXTERNAL_EVIDENCE_DIR)
  get_filename_component(
    HBFSIM_EXTERNAL_EVIDENCE_DIR_ABSOLUTE
    "${HBFSIM_EXTERNAL_EVIDENCE_DIR}"
    ABSOLUTE
    BASE_DIR "${CMAKE_SOURCE_DIR}"
  )
  list(APPEND HBFSIM_EXTERNAL_REPORT_EVIDENCE_ARGS
    --evidence-dir "${HBFSIM_EXTERNAL_EVIDENCE_DIR_ABSOLUTE}"
  )
  list(APPEND HBFSIM_CERTIFICATE_EXTERNAL_EVIDENCE_ARGS
    --external-evidence-dir "${HBFSIM_EXTERNAL_EVIDENCE_DIR_ABSOLUTE}"
  )
endif()

find_package(Git QUIET)
add_custom_target(hbfsim_build_provenance
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_ROOT=${CMAKE_SOURCE_DIR}"
    "-DOUTPUT_DIR=${CMAKE_BINARY_DIR}/generated"
    "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    -P "${CMAKE_SOURCE_DIR}/cmake/BuildProvenance.cmake"
  BYPRODUCTS
    "${CMAKE_BINARY_DIR}/generated/hbfsim_build_provenance.hpp"
    "${CMAKE_BINARY_DIR}/generated/hbfsim_source_manifest.txt"
  VERBATIM
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/Targets.cmake)

if(BUILD_TESTING)
  set(HBFSIM_SERVELOOP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../ServeLoop" CACHE PATH
      "Compatible ServeLoop checkout used by frontend integration tests")
  # Oldest ServeLoop compiler revision whose batches the current hbfsim_client
  # accepts (earlier revisions still emit span_start). This is the only place
  # the minimum is defined; SERVELOOP_REVISION in .github/workflows/ci.yml
  # must match it.
  set(HBFSIM_SERVELOOP_MIN_REVISION 585c563 CACHE STRING
      "Oldest ServeLoop revision the frontend integration tests accept")
  include(cmake/TestTargets.cmake)
  include(cmake/tests/Foundation.cmake)
  include(cmake/tests/Tools.cmake)
  include(cmake/tests/Devices.cmake)
  include(cmake/tests/Simulation.cmake)
  include(cmake/tests/Examples.cmake)

  # Tests that import hbserve, read SERVELOOP_ROOT or use the ../ServeLoop
  # path carry the 'serveloop' label (ci.yml runs -L / -LE serveloop) and
  # require the fixture below, which fails with one line when the checkout is
  # missing or older than HBFSIM_SERVELOOP_MIN_REVISION; the dependent tests
  # are then reported Not Run instead of failing one by one.
  add_test(
    NAME serveloop_checkout_compatible
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tests/python/check_serveloop_checkout.py"
      --root "${HBFSIM_SERVELOOP_SOURCE_DIR}"
      --minimum-revision "${HBFSIM_SERVELOOP_MIN_REVISION}"
  )
  set_tests_properties(serveloop_checkout_compatible PROPERTIES
    FIXTURES_SETUP serveloop
    LABELS serveloop
    TIMEOUT 30
  )
  set(HBFSIM_SERVELOOP_TESTS
    serveloop_frontend_contracts
  )
  set_property(TEST ${HBFSIM_SERVELOOP_TESTS} APPEND PROPERTY
    LABELS serveloop)
  set_property(TEST ${HBFSIM_SERVELOOP_TESTS} APPEND PROPERTY
    FIXTURES_REQUIRED serveloop)

  # Direct script tests and their subprocesses need the same source roots.
  # Keep HBFSim's client first when the frontend checkout carries a copy.
  get_property(HBFSIM_REGISTERED_TESTS DIRECTORY PROPERTY TESTS)
  if(WIN32)
    set(HBFSIM_PYTHONPATH_SEPARATOR "\\;")
  else()
    set(HBFSIM_PYTHONPATH_SEPARATOR ":")
  endif()
  set(HBFSIM_TEST_PYTHONPATH "${CMAKE_SOURCE_DIR}")
  if(EXISTS "${HBFSIM_SERVELOOP_SOURCE_DIR}/hbserve/__init__.py")
    string(APPEND HBFSIM_TEST_PYTHONPATH
      "${HBFSIM_PYTHONPATH_SEPARATOR}${HBFSIM_SERVELOOP_SOURCE_DIR}")
    # Resolve experiment model/config assets from the same frontend checkout.
    set_property(TEST ${HBFSIM_REGISTERED_TESTS} APPEND PROPERTY ENVIRONMENT
      "SERVELOOP_ROOT=${HBFSIM_SERVELOOP_SOURCE_DIR}")
  endif()
  if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
    string(APPEND HBFSIM_TEST_PYTHONPATH
      "${HBFSIM_PYTHONPATH_SEPARATOR}$ENV{PYTHONPATH}")
  endif()
  set_property(TEST ${HBFSIM_REGISTERED_TESTS} APPEND PROPERTY ENVIRONMENT
    "PYTHONPATH=${HBFSIM_TEST_PYTHONPATH}")
endif()
