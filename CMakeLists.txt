cmake_minimum_required(VERSION 3.20)
project(HBFSimPhysical VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(HBFSIM_NODE_EXECUTABLE NAMES node nodejs)

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
set(HBFSIM_GIT_COMMIT "unknown")
set(HBFSIM_GIT_DIRTY 0)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" rev-parse --verify HEAD
    OUTPUT_VARIABLE HBFSIM_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE HBFSIM_GIT_REV_RESULT
  )
  if(NOT HBFSIM_GIT_REV_RESULT EQUAL 0)
    set(HBFSIM_GIT_COMMIT "unknown")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}"
      status --porcelain --untracked-files=no
    OUTPUT_VARIABLE HBFSIM_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT HBFSIM_GIT_STATUS STREQUAL "")
    set(HBFSIM_GIT_DIRTY 1)
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hbfsim_physical
  src/physical/address_heatmap.cpp
  src/physical/external/external_backing_device.cpp
  src/physical/hbm/hbm_device.cpp
  src/physical/hbf/hbf_device.cpp
  src/physical/hybrid/behavioral_tiering_composition.cpp
  src/physical/hybrid/composition_common.cpp
  src/physical/hybrid/capacity_overflow_composition.cpp
  src/physical/hybrid/direct_composition.cpp
  src/physical/hybrid/layer_streaming_composition.cpp
)

target_include_directories(hbfsim_physical PUBLIC src)

target_compile_options(hbfsim_physical PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
)

add_executable(physical_probe
  src/tools/physical_probe.cpp
)

target_link_libraries(physical_probe PRIVATE hbfsim_physical)

add_executable(validation_probe
  src/tools/validation_probe.cpp
)

target_link_libraries(validation_probe PRIVATE hbfsim_physical)

add_executable(scenario_compare
  src/tools/scenario_compare.cpp
)

target_link_libraries(scenario_compare PRIVATE hbfsim_physical)
target_compile_definitions(scenario_compare PRIVATE
  HBFSIM_VERSION="${PROJECT_VERSION}-dev"
  HBFSIM_GIT_COMMIT="${HBFSIM_GIT_COMMIT}"
  HBFSIM_GIT_DIRTY=${HBFSIM_GIT_DIRTY}
  HBFSIM_BUILD_TYPE="$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unspecified>"
  HBFSIM_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  HBFSIM_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

add_executable(overflow_offload_experiment
  src/tools/overflow_offload_experiment.cpp
)
target_link_libraries(overflow_offload_experiment PRIVATE hbfsim_physical)
target_compile_definitions(overflow_offload_experiment PRIVATE
  HBFSIM_VERSION="${PROJECT_VERSION}-dev"
  HBFSIM_GIT_COMMIT="${HBFSIM_GIT_COMMIT}"
  HBFSIM_GIT_DIRTY=${HBFSIM_GIT_DIRTY}
  HBFSIM_BUILD_TYPE="$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unspecified>"
  HBFSIM_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  HBFSIM_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

add_custom_target(physical_guard
  COMMAND /bin/bash
    "${CMAKE_SOURCE_DIR}/tools/run_physical_guards.sh"
    "$<TARGET_FILE:physical_probe>"
    "${CMAKE_SOURCE_DIR}/out/physical-guards"
  DEPENDS physical_probe
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Running physical scenario guards"
)

add_custom_target(use_cases
  COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/run_use_cases.py"
    "--scenario-compare" "$<TARGET_FILE:scenario_compare>"
    "--out-dir" "${CMAKE_SOURCE_DIR}/out/use-cases"
  DEPENDS scenario_compare physical_probe
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Running the component-level use-case checks"
)

add_custom_target(waf_cases
  COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/run_waf_cases.py"
    "--scenario-compare" "$<TARGET_FILE:scenario_compare>"
    "--out-dir" "${CMAKE_SOURCE_DIR}/out/write-amplification-cases"
  DEPENDS scenario_compare
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Running the write-amplification accounting verification cases"
)

add_custom_target(foundational_validation
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/validation/run_validation.py"
    --probe "$<TARGET_FILE:validation_probe>"
    --case-dir "${CMAKE_SOURCE_DIR}/validation/cases"
    --artifact-dir "${CMAKE_BINARY_DIR}/foundational-validation"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/validation/property_fuzz.py"
    --probe "$<TARGET_FILE:validation_probe>"
    --seeds 32
    --max-requests 50
    --artifact-dir "${CMAKE_BINARY_DIR}/foundational-property-fuzz"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/validation/external_differential.py"
    --manifest "${CMAKE_SOURCE_DIR}/validation/external/tools.json"
    --fixture-dir "${CMAKE_SOURCE_DIR}/validation/external/fixtures"
    --repository "${CMAKE_SOURCE_DIR}"
    --hbfsim-probe "$<TARGET_FILE:validation_probe>"
    ${HBFSIM_EXTERNAL_REPORT_EVIDENCE_ARGS}
    --report "${CMAKE_BINARY_DIR}/foundational-external-report.json"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/validation/run_behavioral_differential.py"
    --scenario-compare "$<TARGET_FILE:scenario_compare>"
    --config
      "${CMAKE_SOURCE_DIR}/configs/scenario_compare/server-4k-hbf4x.cfg"
    --seeds 32
    --operations 48
    --report
      "${CMAKE_BINARY_DIR}/foundational-behavioral-differential.json"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/validation/run_analytical_microbench.py"
    --scenario-compare "$<TARGET_FILE:scenario_compare>"
    --config
      "${CMAKE_SOURCE_DIR}/configs/scenario_compare/usecase-baseline.cfg"
    --repository "${CMAKE_SOURCE_DIR}"
    --artifact-dir
      "${CMAKE_BINARY_DIR}/foundational-analytical-microbench"
    --report
      "${CMAKE_BINARY_DIR}/foundational-analytical-microbench.json"
  DEPENDS validation_probe scenario_compare
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT
    "Running independent ledgers and deterministic foundational properties"
)

add_custom_target(foundational_mutation_validation
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/validation/run_mutation_validation.py"
    --repository "${CMAKE_SOURCE_DIR}"
    --report "${CMAKE_BINARY_DIR}/foundational-mutation-report.json"
    --artifact-dir "${CMAKE_BINARY_DIR}/foundational-mutation-artifacts"
    --parallel 2
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Proving foundational gates kill production-code mutations"
)

add_custom_target(foundational_certificate
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/validation/generate_certificate.py"
    --repository "${CMAKE_SOURCE_DIR}"
    --build-dir "${CMAKE_BINARY_DIR}"
    --scenario-compare "$<TARGET_FILE:scenario_compare>"
    --probe "$<TARGET_FILE:validation_probe>"
    --physical-probe "$<TARGET_FILE:physical_probe>"
    --overflow-experiment "$<TARGET_FILE:overflow_offload_experiment>"
    --mutation-report
      "${CMAKE_BINARY_DIR}/foundational-mutation-report.json"
    ${HBFSIM_CERTIFICATE_EXTERNAL_EVIDENCE_ARGS}
    --artifact-dir
      "${CMAKE_BINARY_DIR}/foundational-certificate-artifacts"
    --output "${CMAKE_BINARY_DIR}/foundational-validation-certificate.json"
  DEPENDS
    scenario_compare
    validation_probe
    physical_probe
    overflow_offload_experiment
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Issuing a commit- and binary-bound foundational certificate"
)

if(BUILD_TESTING)
  add_executable(address_heatmap_test
    tests/address_heatmap_test.cpp
  )
  target_link_libraries(address_heatmap_test PRIVATE hbfsim_physical)

  add_executable(closed_loop_window_test
    tests/closed_loop_window_test.cpp
  )
  target_link_libraries(closed_loop_window_test PRIVATE hbfsim_physical)

  add_executable(external_backing_device_test
    tests/external_backing_device_test.cpp
  )
  target_link_libraries(external_backing_device_test PRIVATE hbfsim_physical)

  add_executable(hbf_compact_mutable_test
    tests/hbf_compact_mutable_test.cpp
  )
  target_link_libraries(hbf_compact_mutable_test PRIVATE hbfsim_physical)

  add_executable(capacity_overflow_composition_test
    tests/capacity_overflow_composition_test.cpp
  )
  target_link_libraries(
    capacity_overflow_composition_test PRIVATE hbfsim_physical)

  add_executable(behavioral_tiering_composition_test
    tests/behavioral_tiering_composition_test.cpp
  )
  target_link_libraries(
    behavioral_tiering_composition_test PRIVATE hbfsim_physical)

  set(HBFSIM_TEST_OUT "${CMAKE_BINARY_DIR}/test-output")
  file(MAKE_DIRECTORY "${HBFSIM_TEST_OUT}")

  add_test(
    NAME behavioral_tiering_composition
    COMMAND behavioral_tiering_composition_test
  )

  add_test(
    NAME parameter_provenance_coverage
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tools/check_parameter_provenance.py"
      --config-dir "${CMAKE_SOURCE_DIR}/configs/scenario_compare"
      --registry "${CMAKE_SOURCE_DIR}/configs/parameter-provenance.json"
  )

  add_test(
    NAME foundational_validation_contracts
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/validation/test_validation_contracts.py"
  )

  add_test(
    NAME foundational_independent_oracles
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/validation/run_validation.py"
      --probe "$<TARGET_FILE:validation_probe>"
      --case-dir "${CMAKE_SOURCE_DIR}/validation/cases"
      --artifact-dir "${HBFSIM_TEST_OUT}/foundational-validation"
  )

  add_test(
    NAME foundational_property_fuzz
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/validation/property_fuzz.py"
      --probe "$<TARGET_FILE:validation_probe>"
      --seeds 32
      --max-requests 50
      --artifact-dir "${HBFSIM_TEST_OUT}/foundational-property-fuzz"
  )

  add_test(
    NAME foundational_external_contracts
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/validation/test_external_differential.py"
  )

  add_test(
    NAME foundational_behavioral_differential
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/validation/run_behavioral_differential.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
      --config
        "${CMAKE_SOURCE_DIR}/configs/scenario_compare/server-4k-hbf4x.cfg"
      --seeds 12
      --operations 28
      --report
        "${HBFSIM_TEST_OUT}/foundational-behavioral-differential.json"
  )

  add_test(
    NAME foundational_analytical_microbench
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/validation/run_analytical_microbench.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
      --config
        "${CMAKE_SOURCE_DIR}/configs/scenario_compare/usecase-baseline.cfg"
      --repository "${CMAKE_SOURCE_DIR}"
      --artifact-dir
        "${HBFSIM_TEST_OUT}/foundational-analytical-microbench"
      --report
        "${HBFSIM_TEST_OUT}/foundational-analytical-microbench.json"
  )

  add_test(
    NAME foundational_certificate_contracts
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/validation/test_certificate.py"
  )

  add_test(
    NAME experiment_certificate_gates
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_experiment_certificate_gates.py"
  )

  add_test(
    NAME ec_sanity_census
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_check_ec_sanity.py"
  )

  add_test(
    NAME experiment_config_contracts
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_experiment_configs.py"
  )

  add_test(
    NAME behavioral_workload_quality_contracts
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_assess_workload_quality.py"
  )

  add_test(
    NAME behavioral_placement_experiment_contracts
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_run_behavioral_placement_experiments.py"
  )

  add_test(
    NAME astra_replay_artifact_transaction
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_replay_astra_trace.py"
  )

  add_test(
    NAME qwen_bailian_frontier_adapter
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_prepare_qwen_bailian_workload.py"
  )

  add_test(
    NAME h100_llama2_70b_public_timing_anchor
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_h100_llama2_70b_anchor.py"
  )

  add_test(
    NAME h100_llama2_70b_public_timing_calibration
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_h100_llama2_70b_calibration.py"
  )

  add_test(
    NAME frontier_integration_bundle
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_prepare_frontier_integration.py"
  )

  add_test(
    NAME frontier_hybrid_residency_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_frontier_hybrid_residency.py"
  )

  add_test(
    NAME frontier_replay_accounting_audit
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_audit_frontier_replay.py"
  )

  add_test(
    NAME frontier_memory_object_export
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_export_frontier_memory_trace.py"
  )

  add_test(
    NAME frontier_70b_structural_suite_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_run_frontier_70b_structural_suite.py"
  )

  add_test(
    NAME frontier_70b_capacity_grid_preflight
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_prepare_frontier_70b_capacity_grid.py"
  )

  add_test(
    NAME frontier_70b_two_axis_slice_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_frontier_70b_two_axis_slice.py"
  )

  add_test(
    NAME frontier_70b_two_axis_evidence_certificate
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_verify_frontier_70b_two_axis_evidence.py"
  )

  add_test(
    NAME kimi_k3_capacity_preflight_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_prepare_kimi_k3_capacity_preflight.py"
  )

  add_test(
    NAME frontier_residency_object_binding
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_prepare_frontier_residency_config.py"
  )

  add_test(
    NAME frontier_memory_structural_integration
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_frontier_memory_structural_integration.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )

  add_test(
    NAME frontier_canonical_waf_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_frontier_waf.py"
  )

  add_test(
    NAME frontier_70b_hbf_wear_study_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_frontier_70b_hbf_wear_study.py"
  )
  set_tests_properties(frontier_memory_structural_integration PROPERTIES TIMEOUT 60)

  add_test(
    NAME summary_hbf_stage_work_observability
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_summary_wait_stats.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )

  add_test(
    NAME summary_hybrid_residency_observability
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_summary_layer_streaming_stats.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )

  add_test(
    NAME summary_behavioral_tiering_v16_observability
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_summary_behavioral_tiering.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
      --config
        "${CMAKE_SOURCE_DIR}/configs/scenario_compare/server-4k-hbf4x.cfg"
  )

  add_test(
    NAME lazy_sequential_equivalence_and_option_scoping
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_lazy_sequential_equivalence.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )

  add_test(
    NAME summary_time_breakdown_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_time_breakdown.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )

  add_test(
    NAME time_breakdown_report_renderer
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_time_breakdown_report.py"
  )

  add_test(
    NAME time_breakdown_visualization_renderer
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_plot_time_breakdown.py"
  )

  add_test(
    NAME waf_verifier_mutation_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_run_waf_cases.py"
  )

  add_test(
    NAME gc_waf_quick_accounting
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_gc_waf_quick.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )
  set_tests_properties(gc_waf_quick_accounting PROPERTIES TIMEOUT 60)

  add_test(
    NAME address_heatmap_accounting
    COMMAND address_heatmap_test
  )

  add_test(
    NAME address_heatmap_renderer
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_plot_address_heatmap.py"
  )

  add_test(
    NAME synthetic_trace_generator
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_generate_synthetic_trace.py"
  )

  add_test(
    NAME trace_locality_analyzer
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_analyze_trace_locality.py"
  )

  add_test(
    NAME trace_span_census_scalability
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_trace_span_census.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )

  add_test(
    NAME trace_logical_heatmap
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_plot_trace_heatmap.py"
  )

  add_test(
    NAME synthetic_experiment_plan
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_run_synthetic_experiments.py"
  )

  add_test(
    NAME synthetic_output_smoke_end_to_end
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/run_synthetic_experiments.py"
      --profile smoke
      --groups output
      --out-dir "${HBFSIM_TEST_OUT}/synthetic-output-smoke"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
      --jobs 2
  )
  set_tests_properties(synthetic_output_smoke_end_to_end PROPERTIES
    TIMEOUT 120
  )

  add_test(
    NAME address_heatmap_cpp_python_interop
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_address_heatmap_interop.py"
      --emitter "$<TARGET_FILE:address_heatmap_test>"
  )

  if(HBFSIM_NODE_EXECUTABLE)
    add_test(
      NAME architecture_design_sync
      COMMAND "${HBFSIM_NODE_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/web/architecture-editor/tools/design-sync.mjs"
    )
  endif()

  add_test(
    NAME scenario_compare_version
    COMMAND scenario_compare --version
  )
  set_tests_properties(scenario_compare_version PROPERTIES
    PASS_REGULAR_EXPRESSION "HBFSim scenario_compare ${PROJECT_VERSION}-dev"
  )

  add_test(
    NAME physical_probe_all
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
    NAME hbf_compact_mutable_initial_image
    COMMAND hbf_compact_mutable_test
  )

  add_test(
    NAME external_backing_layer_streaming_contract
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_external_backing.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
  )
  set_tests_properties(
    external_backing_layer_streaming_contract PROPERTIES TIMEOUT 60)

  add_test(
    NAME capacity_overflow_composition_contract
    COMMAND capacity_overflow_composition_test
  )

  add_test(
    NAME capacity_overflow_experiment_end_to_end
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/test_capacity_overflow_experiment.py"
      --experiment "$<TARGET_FILE:overflow_offload_experiment>"
  )
  set_tests_properties(
    capacity_overflow_experiment_end_to_end PROPERTIES TIMEOUT 330)

  add_test(
    NAME use_cases_quick_contracts
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_SOURCE_DIR}/tools/run_use_cases.py"
      --scenario-compare "$<TARGET_FILE:scenario_compare>"
      --out-dir "${HBFSIM_TEST_OUT}/use-cases-quick"
      --skip-deep-waf
  )
  set_tests_properties(use_cases_quick_contracts PROPERTIES TIMEOUT 300)

  add_test(
    NAME scenario_generate_llm_trace
    COMMAND scenario_compare
      --generate-semantic-llm "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --llm-tokens 4
      --llm-layers 2
  )

  add_test(
    NAME scenario_accepts_wide_hbf_subarray_calendar
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 128
      --scenarios all-HBF
      --hbf-subarrays-per-plane 96
  )
  set_tests_properties(scenario_accepts_wide_hbf_subarray_calendar PROPERTIES
    DEPENDS scenario_generate_llm_trace
  )

  add_test(
    NAME scenario_reject_overflowing_llm_generator
    COMMAND scenario_compare
      --generate-llm "${HBFSIM_TEST_OUT}/overflow.ramulator.trace"
      --llm-tokens 1
      --llm-layers 1
      --llm-weight-base 18446744073709551615
  )
  set_tests_properties(scenario_reject_overflowing_llm_generator PROPERTIES
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_compare_policy_set
    COMMAND scenario_compare
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
      --hbm-trcdrd-ns 15.25
      --hbm-trcdwr-ns 15.5
      --hbm-tcl-ns 15.75
      --hbm-tcwl-ns 11.25
      --hbm-trp-ns 15.125
      --hbm-tras-ns 34.25
      --hbm-trc-ns 49.5
      --hbm-twr-ns 16.25
      --hbm-trtp-ns 8.25
      --hbm-tccd-s-cycles 4
      --hbm-tccd-l-cycles 8
      --hbm-trrd-s-ns 4.5
      --hbm-trrd-l-ns 6.5
      --hbm-tfaw-ns 21.25
      --hbm-twtr-s-ns 4.5
      --hbm-twtr-l-ns 8.5
      --hbm-trtw-ns 8.75
      --hbf-page-size 4096
      --hbf-oob-bytes 224
      --hbf-channels 4
      --hbf-dies-per-channel 4
      --hbf-ecc-decode-raw-bw 105.46875
      --hbf-ecc-encode-raw-bw 105.46875
      --hbf-channel-bw 421.875
      --hbf-hbio-bw 1600
      --hbf-tsv-bw 1712.5
      --summary-csv "${HBFSIM_TEST_OUT}/policy-set.csv"
      --summary-json "${HBFSIM_TEST_OUT}/policy-set.json"
      --config-out "${HBFSIM_TEST_OUT}/policy-set.cfg"
  )
  set_tests_properties(scenario_compare_policy_set PROPERTIES
    DEPENDS scenario_generate_llm_trace
  )

  set(HBFSIM_PUBLISHED_SCENARIO_CONFIGS
    server-4k-hbf4x-large-layer-buffer.cfg
    server-4k-hbf4x-tight-fabric.cfg
    server-4k-hbf4x.cfg
    server-4k-hbf8x.cfg
    usecase-2h6f.cfg
    usecase-4h4f.cfg
    usecase-6h2f.cfg
    usecase-baseline.cfg
    usecase-baseline-local-output.cfg
    usecase-baseline-output-upper-bound.cfg
  )
  foreach(HBFSIM_CONFIG IN LISTS HBFSIM_PUBLISHED_SCENARIO_CONFIGS)
    string(REPLACE ".cfg" "" HBFSIM_CONFIG_NAME "${HBFSIM_CONFIG}")
    add_test(
      NAME "published_config_${HBFSIM_CONFIG_NAME}"
      COMMAND scenario_compare
        --config "${CMAKE_SOURCE_DIR}/configs/scenario_compare/${HBFSIM_CONFIG}"
        --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
        --max-ops 1
        --scenarios all-HBM
    )
    set_tests_properties("published_config_${HBFSIM_CONFIG_NAME}" PROPERTIES
      DEPENDS scenario_generate_llm_trace
    )
  endforeach()

  foreach(HBFSIM_OUTPUT_CONFIG IN ITEMS
      usecase-baseline-local-output.cfg
      usecase-baseline-output-upper-bound.cfg)
    string(REPLACE ".cfg" "" HBFSIM_OUTPUT_CONFIG_NAME
      "${HBFSIM_OUTPUT_CONFIG}")
    add_test(
      NAME "output_config_${HBFSIM_OUTPUT_CONFIG_NAME}_hbf_path"
      COMMAND scenario_compare
        --config
          "${CMAKE_SOURCE_DIR}/configs/scenario_compare/${HBFSIM_OUTPUT_CONFIG}"
        --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
        --max-ops 1
        --scenarios HBF-static-direct-read
        --static-direct-hbm-bytes 0
    )
    set_tests_properties(
      "output_config_${HBFSIM_OUTPUT_CONFIG_NAME}_hbf_path" PROPERTIES
      DEPENDS scenario_generate_llm_trace
    )
  endforeach()

  add_test(
    NAME scenario_compare_config_replay
    COMMAND scenario_compare
      --config "${HBFSIM_TEST_OUT}/policy-set.cfg"
      --summary-csv "${HBFSIM_TEST_OUT}/policy-set.replay.csv"
      --summary-json "${HBFSIM_TEST_OUT}/policy-set.replay.json"
  )
  set_tests_properties(scenario_compare_config_replay PROPERTIES
    DEPENDS scenario_compare_policy_set
  )

  add_test(
    NAME scenario_compare_capacity_ratio
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 512
      --hbm-capacity-bytes 2147483648
      --flat-hbm-bytes 1073741824
      --hbf-capacity-ratio 4
      --layer-buffer-bytes 33554432
      --summary-json "${HBFSIM_TEST_OUT}/capacity-ratio.json"
  )
  set_tests_properties(scenario_compare_capacity_ratio PROPERTIES
    DEPENDS scenario_generate_llm_trace
  )

  add_test(
    NAME scenario_compare_replay_csv_matches
    COMMAND ${CMAKE_COMMAND} -E compare_files
      "${HBFSIM_TEST_OUT}/policy-set.csv"
      "${HBFSIM_TEST_OUT}/policy-set.replay.csv"
  )
  set_tests_properties(scenario_compare_replay_csv_matches PROPERTIES
    DEPENDS scenario_compare_config_replay
  )

  add_test(
    NAME scenario_compare_replay_json_matches
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tools/compare_summary_results.py"
      "${HBFSIM_TEST_OUT}/policy-set.json"
      "${HBFSIM_TEST_OUT}/policy-set.replay.json"
  )
  set_tests_properties(scenario_compare_replay_json_matches PROPERTIES
    DEPENDS scenario_compare_config_replay
  )

  add_test(
    NAME scenario_reject_oversized_hbf_page
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 64
      --hbf-page-size 8192
  )
  set_tests_properties(scenario_reject_oversized_hbf_page PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_inconsistent_hbm_interface
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBM
      --hbm-channel-width-bits 65
  )
  set_tests_properties(scenario_reject_inconsistent_hbm_interface PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_fractional_hbm_tccd_cycles
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBM
      --hbm-tccd-s-cycles 2.5
  )
  set_tests_properties(scenario_reject_fractional_hbm_tccd_cycles PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_fractional_hbm_clock_ratio
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBM
      --hbm-data-rate-per-command-clock 3.5
  )
  set_tests_properties(scenario_reject_fractional_hbm_clock_ratio PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_ambiguous_hbm_bank_group_scope
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBM
      --hbm-bank-groups 16
  )
  set_tests_properties(scenario_reject_ambiguous_hbm_bank_group_scope PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_removed_hbm_bandwidth_override
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBM
      --hbm-channel-bw 51.2
  )
  set_tests_properties(scenario_reject_removed_hbm_bandwidth_override PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_zero_hbf_ecc_bandwidth
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBF
      --hbf-ecc-decode-raw-bw 0
  )
  set_tests_properties(scenario_reject_zero_hbf_ecc_bandwidth PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_nonfinite_hbf_ecc_latency
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBF
      --hbf-ecc-encode-latency-ns nan
  )
  set_tests_properties(scenario_reject_nonfinite_hbf_ecc_latency PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_hbf_ecc_latency_shorter_than_codeword
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 1
      --scenarios all-HBM
      --hbf-ecc-decode-latency-ns 1
  )
  set_tests_properties(
    scenario_reject_hbf_ecc_latency_shorter_than_codeword PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )

  add_test(
    NAME scenario_reject_empty_layer_buffer
    COMMAND scenario_compare
      --trace "${HBFSIM_TEST_OUT}/llm-smoke.ramulator.trace"
      --max-ops 64
      --layer-buffer-bytes 0
  )
  set_tests_properties(scenario_reject_empty_layer_buffer PROPERTIES
    DEPENDS scenario_generate_llm_trace
    WILL_FAIL TRUE
  )
endif()
