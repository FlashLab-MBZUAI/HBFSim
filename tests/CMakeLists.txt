add_executable(obelisk_tests
    unit/test_subarray.cc
    unit/test_scheduler.cc
    unit/test_address_mapper.cc
    unit/test_trace_reader.cc
    unit/test_end_to_end.cc
)

target_link_libraries(obelisk_tests PRIVATE
    obelisk_core
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(obelisk_tests)
