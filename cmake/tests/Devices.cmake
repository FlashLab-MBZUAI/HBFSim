add_test(
  NAME hbfsim_version
  COMMAND hbfsim --version
)
set_tests_properties(hbfsim_version PROPERTIES
  PASS_REGULAR_EXPRESSION "HBFSim engine ${PROJECT_VERSION}-dev"
)

add_test(
  NAME hbfsim_reference_version
  COMMAND hbfsim_reference --version
)
set_tests_properties(hbfsim_reference_version PROPERTIES
  PASS_REGULAR_EXPRESSION "HBFSim ${PROJECT_VERSION}-dev"
)

add_test(
  NAME physical_invariants
  COMMAND physical_probe all
)

add_test(
  NAME hbm_interface_frequency_width_contract
  COMMAND physical_probe hbm-interface
)

add_test(
  NAME hbf_ecc_pipeline_contract
  COMMAND physical_probe hbf-ecc-pipeline
)

add_test(
  NAME closed_loop_window_completion_order
  COMMAND closed_loop_window_test
)

add_test(
  NAME external_backing_device_contract
  COMMAND external_backing_device_test
)

add_test(
  NAME external_hardware_calibration_replay_self_test
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/evidence/hardware/dana_a100_offload/calibrate.py"
    --self-test
    --probe "$<TARGET_FILE:external_calibration_probe>"
)

add_test(
  NAME hbf_compact_mutable_initial_image
  COMMAND hbf_compact_mutable_test
)

add_test(
  NAME hbf_persistent_image_restore
  COMMAND hbf_persistent_image_test
)

add_test(
  NAME hbf_mapping_cache
  COMMAND hbf_mapping_cache_test
)
add_test(NAME hbf_mapping_layouts COMMAND hbf_mapping_layout_test)

add_test(
  NAME hbf_thermal_governor
  COMMAND hbf_thermal_test
)

add_test(NAME hbf_read_buffer_handoff COMMAND hbf_read_buffer_handoff_test)
add_test(NAME hbf_read_buffer_pressure COMMAND hbf_read_buffer_pressure_test)
set_tests_properties(hbf_read_buffer_handoff hbf_read_buffer_pressure
  PROPERTIES TIMEOUT 30)

add_test(
  NAME hbf_gc_cached_mapping_progress
  COMMAND hbf_gc_reserve_test
)
set_tests_properties(hbf_gc_cached_mapping_progress PROPERTIES TIMEOUT 60)

add_test(NAME hbf_checkpoint_pressure
  COMMAND hbf_checkpoint_pressure_test "${CMAKE_SOURCE_DIR}/configs/systems/4hbm-4hbf.cfg")
set_tests_properties(hbf_checkpoint_pressure PROPERTIES TIMEOUT 60)

add_test(NAME hbf_gc_index_equivalence COMMAND hbf_gc_index_test)
set_tests_properties(hbf_gc_index_equivalence PROPERTIES TIMEOUT 60)

add_test(NAME hbf_gc_drain_parallelism COMMAND hbf_gc_drain_parallel_test)
set_tests_properties(hbf_gc_drain_parallelism PROPERTIES TIMEOUT 60)

add_test(
  NAME hbf_static_wear_leveling
  COMMAND hbf_static_wear_leveling_test
)
set_tests_properties(hbf_static_wear_leveling PROPERTIES TIMEOUT 120)



add_test(NAME hbm_clock_roundoff COMMAND hbm_clock_roundoff_test)
set_tests_properties(hbm_clock_roundoff PROPERTIES TIMEOUT 30)

add_test(NAME hbm_controller_buffer COMMAND hbm_controller_buffer_test)
add_test(NAME gap_calendar_equivalence COMMAND gap_calendar_test)
set_tests_properties(gap_calendar_equivalence PROPERTIES TIMEOUT 120)

add_test(
  NAME external_backing_layer_streaming_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_external_backing.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
)
set_tests_properties(
  external_backing_layer_streaming_contract PROPERTIES TIMEOUT 60)

add_test(
  NAME component_contracts
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/components.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --out-dir "${HBFSIM_TEST_OUT}/component-checks"
)
set_tests_properties(component_contracts PROPERTIES TIMEOUT 300)

add_test(NAME hbf_host_zone_remapping COMMAND hbf_host_zones_test)
add_test(NAME hbf_mapping_organizations COMMAND hbf_mapping_organization_test)

add_test(NAME ocp_standard_contract COMMAND ocp_standard_test)
set_tests_properties(ocp_standard_contract PROPERTIES TIMEOUT 30)

add_test(NAME hbm_channel_model COMMAND hbm_channel_model_test)
