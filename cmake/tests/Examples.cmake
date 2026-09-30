# The `python3 -m hbfsim` front door and every script under examples/.
# Each example runs into a directory that a setup fixture empties first.

add_test(
  NAME hbfsim_frontdoor_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_hbfsim_frontdoor.py"
    --build-dir "${CMAKE_BINARY_DIR}"
)
set(HBFSIM_FRONTDOOR_TESTS hbfsim_frontdoor_contract)

file(GLOB HBFSIM_EXAMPLES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/examples/[0-9][0-9]_*.py")
foreach(HBFSIM_EXAMPLE IN LISTS HBFSIM_EXAMPLES)
  get_filename_component(HBFSIM_EXAMPLE_NAME "${HBFSIM_EXAMPLE}" NAME_WE)
  set(HBFSIM_EXAMPLE_OUT "${HBFSIM_TEST_OUT}/examples/${HBFSIM_EXAMPLE_NAME}")
  add_test(
    NAME "example_${HBFSIM_EXAMPLE_NAME}_clean"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${HBFSIM_EXAMPLE_OUT}"
  )
  set_tests_properties("example_${HBFSIM_EXAMPLE_NAME}_clean" PROPERTIES
    FIXTURES_SETUP "example_${HBFSIM_EXAMPLE_NAME}"
  )
  add_test(
    NAME "example_${HBFSIM_EXAMPLE_NAME}"
    COMMAND "${Python3_EXECUTABLE}" -B "${HBFSIM_EXAMPLE}"
      --out "${HBFSIM_EXAMPLE_OUT}"
  )
  set_tests_properties("example_${HBFSIM_EXAMPLE_NAME}" PROPERTIES
    FIXTURES_REQUIRED "example_${HBFSIM_EXAMPLE_NAME}"
  )
  list(APPEND HBFSIM_FRONTDOOR_TESTS "example_${HBFSIM_EXAMPLE_NAME}")
endforeach()

set_property(TEST ${HBFSIM_FRONTDOOR_TESTS} APPEND PROPERTY ENVIRONMENT
  "HBFSIM_BUILD_DIR=${CMAKE_BINARY_DIR}"
)
