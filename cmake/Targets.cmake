# One warning set for every compiled target (libraries, executables, probes,
# research drivers and the CTest executables included after this file from
# the same directory scope). CI adds -Werror on top of it.
#
# -Wno-missing-field-initializers: the physical model and the host controller
# use designated initializers that deliberately name only the members that
# differ from the default, e.g. PendingBlockTransition{.finish_ns = ...};
# the omitted members are value-initialized by the language. Both GCC and
# Clang report such initializers under -Wextra's -Wmissing-field-initializers,
# so that single check is disabled instead of padding every initializer.
set(HBFSIM_WARNING_OPTIONS
  -Wall
  -Wextra
  -Wpedantic
  -Wno-missing-field-initializers
)
add_compile_options(
  "$<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:${HBFSIM_WARNING_OPTIONS}>"
)

add_library(hbfsim_core
  src/physical/address_heatmap.cpp
  src/physical/base_die_link.cpp
  src/physical/external/cxl_ssd.cpp
  src/physical/external/external_backing_device.cpp
  src/physical/hbm/hbm_device.cpp
  src/physical/hbf/hbf_device.cpp
  src/physical/resource_calendar.cpp
  src/host/hbf_gc.cpp
  src/host/hbf_zones.cpp
  src/host/hbf_zone_endurance.cpp
  src/host/hbf_invalidation.cpp
  src/host/hbf_wear_snapshot.cpp
  src/host/hbf_controller.cpp
  src/host/hbf_mapping_policy.cpp
  src/host/hbf_persistent_image.cpp
  src/physical/simulation_session.cpp
)

target_include_directories(hbfsim_core PUBLIC src)

add_library(hbfsim_policy_support
  src/policies/policy_common.cpp
)
target_include_directories(hbfsim_policy_support PUBLIC src)
target_link_libraries(hbfsim_policy_support PUBLIC hbfsim_core)

# Reference policies are deliberately outside the simulator core. They are
# reusable experiment implementations, not the simulator's fixed scenario
# surface. Arbitrary policy controllers use SimulationSession directly.
add_library(hbfsim_reference_policies
  src/policies/reference/behavioral_tiering_policy.cpp
  src/policies/reference/direct_policy.cpp
  src/policies/reference/layer_streaming_policy.cpp
)
target_include_directories(hbfsim_reference_policies PUBLIC src)
target_link_libraries(hbfsim_reference_policies PUBLIC hbfsim_policy_support)

add_executable(physical_probe
  verification/probes/physical.cpp
)

target_link_libraries(physical_probe PRIVATE hbfsim_reference_policies)

add_executable(ledger_probe
  verification/probes/ledger.cpp
)

target_link_libraries(ledger_probe PRIVATE hbfsim_reference_policies)

# Hardware-calibration replay: this deliberately links the production
# ExternalBackingDevice instead of duplicating its timing equations in the
# calibration driver.
add_executable(external_calibration_probe
  evidence/hardware/dana_a100_offload/external_calibration_probe.cpp
)

target_link_libraries(external_calibration_probe PRIVATE hbfsim_core)

add_library(hbfsim_engine_app
  src/app/hbf_wear_report.cpp
  src/app/session_protocol.cpp
  src/app/system_config.cpp
)
target_include_directories(hbfsim_engine_app PUBLIC src "${CMAKE_BINARY_DIR}/generated")
add_dependencies(hbfsim_engine_app hbfsim_build_provenance)
target_link_libraries(hbfsim_engine_app PUBLIC hbfsim_core)

add_executable(hbfsim
  src/app/session_main.cpp
)

target_link_libraries(hbfsim PRIVATE hbfsim_engine_app)
target_compile_definitions(hbfsim PRIVATE
  HBFSIM_VERSION="${PROJECT_VERSION}-dev"
  HBFSIM_BUILD_TYPE="$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unspecified>"
  HBFSIM_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  HBFSIM_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

add_executable(hbfsim_reference
  src/app/main.cpp
  src/app/reference_runner.cpp
)
set_target_properties(hbfsim_reference PROPERTIES OUTPUT_NAME hbfsim-reference)

target_link_libraries(hbfsim_reference PRIVATE
  hbfsim_reference_policies
  hbfsim_engine_app
)
target_compile_definitions(hbfsim_reference PRIVATE
  HBFSIM_VERSION="${PROJECT_VERSION}-dev"
  HBFSIM_BUILD_TYPE="$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unspecified>"
  HBFSIM_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  HBFSIM_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

add_custom_target(verify_physical
  COMMAND /bin/bash
    "${CMAKE_SOURCE_DIR}/verification/gates/physical.sh"
    "$<TARGET_FILE:physical_probe>"
    "${CMAKE_SOURCE_DIR}/out/physical-guards"
  DEPENDS physical_probe
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Verifying physical-model invariants"
)

add_custom_target(verify_components
  COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/verification/gates/components.py"
    "--simulator" "$<TARGET_FILE:hbfsim_reference>"
    "--out-dir" "${CMAKE_SOURCE_DIR}/out/component-checks"
  DEPENDS hbfsim_reference
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Verifying component input/output contracts"
)

add_custom_target(verify_write_amplification
  COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/verification/gates/waf.py"
    "--simulator" "$<TARGET_FILE:hbfsim_reference>"
    "--out-dir" "${CMAKE_SOURCE_DIR}/out/write-amplification-cases"
  DEPENDS hbfsim_reference
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Verifying write-amplification accounting and GC regimes"
)

add_custom_target(verify_foundation
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/cases.py"
    --probe "$<TARGET_FILE:ledger_probe>"
    --case-dir "${CMAKE_SOURCE_DIR}/verification/cases"
    --artifact-dir "${CMAKE_BINARY_DIR}/foundational-validation"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/fuzz.py"
    --probe "$<TARGET_FILE:ledger_probe>"
    --seeds 32
    --max-requests 50
    --artifact-dir "${CMAKE_BINARY_DIR}/foundational-property-fuzz"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/evidence/external/verify.py"
    --manifest "${CMAKE_SOURCE_DIR}/evidence/external/assets/tools.json"
    --fixture-dir "${CMAKE_SOURCE_DIR}/evidence/external/assets/fixtures"
    --repository "${CMAKE_SOURCE_DIR}"
    --hbfsim-probe "$<TARGET_FILE:ledger_probe>"
    ${HBFSIM_EXTERNAL_REPORT_EVIDENCE_ARGS}
    --report "${CMAKE_BINARY_DIR}/foundational-external-report.json"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/placement.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --config
      "${CMAKE_SOURCE_DIR}/configs/systems/server-hbm128-hbf512.cfg"
    --seeds 32
    --operations 48
    --report
      "${CMAKE_BINARY_DIR}/foundational-behavioral-differential.json"
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/analytical.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --config
      "${CMAKE_SOURCE_DIR}/configs/systems/eight-stack-baseline.cfg"
    --repository "${CMAKE_SOURCE_DIR}"
    --artifact-dir
      "${CMAKE_BINARY_DIR}/foundational-analytical-microbench"
    --report
      "${CMAKE_BINARY_DIR}/foundational-analytical-microbench.json"
  DEPENDS ledger_probe hbfsim_reference
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT
    "Running independent ledgers and deterministic foundational properties"
)

add_custom_target(verify_mutations
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/mutation.py"
    --repository "${CMAKE_SOURCE_DIR}"
    --report "${CMAKE_BINARY_DIR}/foundational-mutation-report.json"
    --artifact-dir "${CMAKE_BINARY_DIR}/foundational-mutation-artifacts"
    --parallel 2
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Proving foundational gates kill production-code mutations"
)

add_custom_target(issue_certificate
  COMMAND
    "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/certificate.py"
    --repository "${CMAKE_SOURCE_DIR}"
    --build-dir "${CMAKE_BINARY_DIR}"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --probe "$<TARGET_FILE:ledger_probe>"
    --physical-probe "$<TARGET_FILE:physical_probe>"
    --mutation-report
      "${CMAKE_BINARY_DIR}/foundational-mutation-report.json"
    ${HBFSIM_CERTIFICATE_EXTERNAL_EVIDENCE_ARGS}
    --artifact-dir
      "${CMAKE_BINARY_DIR}/foundational-certificate-artifacts"
    --output "${CMAKE_BINARY_DIR}/foundational-validation-certificate.json"
  DEPENDS
    hbfsim_reference
    ledger_probe
    physical_probe
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Issuing a commit- and binary-bound foundational certificate"
)
