add_test(
  NAME core_rejects_reference_policy_config
  COMMAND hbfsim
    --system-config
      "${CMAKE_SOURCE_DIR}/configs/systems/server-hbm128-hbf512.cfg"
    --system-config
      "${CMAKE_SOURCE_DIR}/configs/policies/reference/server-hbm128-hbf512.cfg"
)
set_tests_properties(core_rejects_reference_policy_config PROPERTIES
  WILL_FAIL TRUE
)

add_test(
  NAME semantic_trace_generation
  COMMAND hbfsim_reference
    --generate-semantic-llm "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --llm-tokens 4
    --llm-layers 2
)


add_test(
  NAME reject_overflowing_trace_generation
  COMMAND hbfsim_reference
    --generate-semantic-llm "${HBFSIM_TEST_OUT}/overflow.ramulator.trace"
    --llm-tokens 1
    --llm-layers 1
    --llm-weight-base 18446744073709551615
)
set_tests_properties(reject_overflowing_trace_generation PROPERTIES
  WILL_FAIL TRUE
)

add_test(
  NAME resolved_policy_export
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 512
    --flat-hbm-bytes 1073741824
    --layer-buffer-bytes 32768
    --hbm-channel-row-size-bytes 4096
    --hbm-channel-width-bits 128
    --hbm-burst-length 16
    --hbm-pin-rate-gbps 7.2
    --hbm-data-rate-per-command-clock 4
    --hbm-bank-groups-per-pseudo-channel 8
    --hbm-banks-per-group 3
    --hbm-address-mapping-ns 0.25
    --hbf-page-size 4096
    --hbf-oob-bytes 224
    --hbf-channels 4
    --hbf-dies-per-channel 4
    --hbf-ecc-decode-raw-bw 105.46875
    --hbf-ecc-encode-raw-bw 105.46875
    --hbf-channel-bw 421.875
    --hbf-speed-grade 2
    --hbf-tsv-bw 1712.5
    --summary-csv "${HBFSIM_TEST_OUT}/policy-set.csv"
    --summary-json "${HBFSIM_TEST_OUT}/policy-set.json"
    --config-out "${HBFSIM_TEST_OUT}/policy-set.cfg"
)

add_test(
  NAME core_cached_mapping_config
  COMMAND hbfsim
    --system-config
      "${CMAKE_SOURCE_DIR}/tests/fixtures/simulation-session-mini.cfg"
    --hbf-mapping-mode cached
    --hbf-ctrl-dram-bytes 16400
)
set_tests_properties(core_cached_mapping_config PROPERTIES
  PASS_REGULAR_EXPRESSION "\"hbf_mapping_mode\":\"cached\"")

add_test(
  NAME simulation_session_checkpoint_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_simulation_session_checkpoint.py"
    --simulator "$<TARGET_FILE:hbfsim>"
)

add_test(
  NAME hbf_live_heat_stream_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_hbf_live_heat_stream.py"
    --simulator "$<TARGET_FILE:hbfsim>"
)
set_tests_properties(resolved_policy_export PROPERTIES
  DEPENDS semantic_trace_generation
)

set(HBFSIM_PUBLISHED_SCENARIO_CONFIGS
  server-hbm128-hbf512-large-buffer.cfg
  server-hbm128-hbf512-tight-fabric.cfg
  server-hbm128-hbf512.cfg
  server-hbm128-hbf1024.cfg
  2hbm-6hbf.cfg
  4hbm-4hbf.cfg
  6hbm-2hbf.cfg
  eight-stack-baseline.cfg
)
foreach(HBFSIM_CONFIG IN LISTS HBFSIM_PUBLISHED_SCENARIO_CONFIGS)
  string(REPLACE ".cfg" "" HBFSIM_CONFIG_NAME "${HBFSIM_CONFIG}")
  # The core accepts every published profile. --describe-system resolves and
  # prints the configuration, then exits; without it the engine would read
  # transaction batches from stdin until EOF and hang under a harness whose
  # stdin is an open pipe.
  add_test(
    NAME "core_system_config_${HBFSIM_CONFIG_NAME}"
    COMMAND hbfsim
      --system-config
        "${CMAKE_SOURCE_DIR}/configs/systems/${HBFSIM_CONFIG}"
      --describe-system
  )
  add_test(
    NAME "published_config_${HBFSIM_CONFIG_NAME}"
    COMMAND hbfsim_reference
      --config "${CMAKE_SOURCE_DIR}/configs/systems/${HBFSIM_CONFIG}"
      --config
        "${CMAKE_SOURCE_DIR}/configs/policies/reference/${HBFSIM_CONFIG}"
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-hbm
  )
  set_tests_properties("published_config_${HBFSIM_CONFIG_NAME}" PROPERTIES
    DEPENDS semantic_trace_generation
  )
endforeach()


add_test(
  NAME resolved_config_replay
  COMMAND hbfsim_reference
    --config "${HBFSIM_TEST_OUT}/policy-set.cfg"
    --summary-csv "${HBFSIM_TEST_OUT}/policy-set.replay.csv"
    --summary-json "${HBFSIM_TEST_OUT}/policy-set.replay.json"
)
set_tests_properties(resolved_config_replay PROPERTIES
  DEPENDS resolved_policy_export
)

add_test(
  NAME capacity_ratio_resolution
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 512
    --hbm-capacity-bytes 2147483648
    --flat-hbm-bytes 1073741824
    --hbf-capacity-ratio 4
    --layer-buffer-bytes 33554432
    --summary-json "${HBFSIM_TEST_OUT}/capacity-ratio.json"
)
set_tests_properties(capacity_ratio_resolution PROPERTIES
  DEPENDS semantic_trace_generation
)

add_test(
  NAME resolved_config_csv_equivalence
  COMMAND ${CMAKE_COMMAND} -E compare_files
    "${HBFSIM_TEST_OUT}/policy-set.csv"
    "${HBFSIM_TEST_OUT}/policy-set.replay.csv"
)
set_tests_properties(resolved_config_csv_equivalence PROPERTIES
  DEPENDS resolved_config_replay
)

add_test(
  NAME resolved_config_json_equivalence
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/reports/compare.py"
    "${HBFSIM_TEST_OUT}/policy-set.json"
    "${HBFSIM_TEST_OUT}/policy-set.replay.json"
)
set_tests_properties(resolved_config_json_equivalence PROPERTIES
  DEPENDS resolved_config_replay
)

add_test(
  NAME reject_oversized_hbf_page
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 64
    --hbf-page-size 8192
)
set_tests_properties(reject_oversized_hbf_page PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

add_test(
  NAME reject_inconsistent_hbm_interface
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 1
    --scenarios all-hbm
    --hbm-channel-width-bits 65
)
set_tests_properties(reject_inconsistent_hbm_interface PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

add_test(
  NAME reject_fractional_hbm_quantum
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 1
    --scenarios all-hbm
    --hbm-service-quantum-bytes 2.5
)
set_tests_properties(reject_fractional_hbm_quantum PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

add_test(
  NAME reject_fractional_hbm_clock_ratio
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 1
    --scenarios all-hbm
    --hbm-data-rate-per-command-clock 3.5
)
set_tests_properties(reject_fractional_hbm_clock_ratio PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

add_test(
  NAME reject_zero_hbf_ecc_bandwidth
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 1
    --scenarios all-hbf
    --hbf-ecc-decode-raw-bw 0
)
set_tests_properties(reject_zero_hbf_ecc_bandwidth PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

add_test(
  NAME reject_nonfinite_hbf_ecc_latency
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 1
    --scenarios all-hbf
    --hbf-ecc-encode-latency-ns nan
)
set_tests_properties(reject_nonfinite_hbf_ecc_latency PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

add_test(
  NAME reject_hbf_ecc_latency_shorter_than_codeword
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 1
    --scenarios all-hbm
    --hbf-ecc-decode-latency-ns 1
)
set_tests_properties(
  reject_hbf_ecc_latency_shorter_than_codeword PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

add_test(
  NAME reject_empty_layer_buffer
  COMMAND hbfsim_reference
    --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
    --max-ops 64
    --layer-buffer-bytes 0
)
set_tests_properties(reject_empty_layer_buffer PROPERTIES
  DEPENDS semantic_trace_generation
  WILL_FAIL TRUE
)

# One HBM issue discipline: FLAT routed entirely to HBM equals all-hbm bit
# for bit on the README quick-start trace.
add_test(
  NAME reference_policy_identity
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_reference_policy_identity.py"
    --reference "$<TARGET_FILE:hbfsim_reference>"
)

# One hardware parser: every published system-profile key round-trips
# through --config-out, CLI overrides reach the engine builder, and unowned
# keys are rejected.
add_test(
  NAME reference_config_roundtrip
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_reference_config_roundtrip.py"
    --reference "$<TARGET_FILE:hbfsim_reference>"
)

add_test(NAME hbf_host_zone_session
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_hbf_host_zones.py"
    --simulator "$<TARGET_FILE:hbfsim>")
add_test(NAME hbf_mapping_organization_session
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/python/test_hbf_mapping_organizations.py"
    --simulator "$<TARGET_FILE:hbfsim>")
add_test(NAME simulation_transaction_completions_contract
  COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_SOURCE_DIR}/tests/python/test_simulation_transaction_completions.py")

add_test(NAME hbf_logical_invalidation COMMAND hbf_logical_invalidation_test)
add_test(NAME host_memory_lifecycle COMMAND host_memory_test)
add_test(NAME hbf_logical_invalidation_session
  COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_SOURCE_DIR}/tests/python/test_hbf_logical_invalidation.py"
    --simulator "$<TARGET_FILE:hbfsim>")

add_test(
  NAME cpu_dram_and_ssd_attachment_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_host_dram_attachment.py"
    --simulator "$<TARGET_FILE:hbfsim>"
)

# Every reference scenario accepts every published system/policy pair (the
# list above is parsed from this file), and the core scenarios run cleanly.
add_test(
  NAME published_config_scenario_matrix
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_reference_scenarios_on_published_configs.py"
    --reference "$<TARGET_FILE:hbfsim_reference>"
)
