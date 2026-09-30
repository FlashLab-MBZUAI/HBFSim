add_test(
  NAME behavioral_tiering_policy
  COMMAND behavioral_tiering_policy_test
)

add_test(
  NAME simulation_session_contract
  COMMAND simulation_session_test
)

add_test(
  NAME system_config_ownership_contract
  COMMAND system_config_test
)

add_test(
  NAME simulation_session_protocol_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_simulation_session_protocol.py"
    --simulator "$<TARGET_FILE:hbfsim>"
)

add_test(
  NAME simulation_session_provenance_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_provenance.py"
)

add_test(
  NAME build_source_provenance_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_build_provenance.py"
    --cmake "${CMAKE_COMMAND}"
    --compiler "${CMAKE_CXX_COMPILER}"
)

add_test(
  NAME parameter_provenance_coverage
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/verification/gates/provenance.py"
    --config-root "${CMAKE_SOURCE_DIR}/configs"
    --registry "${CMAKE_SOURCE_DIR}/configs/parameter-provenance.json"
)

add_test(
  NAME documentation_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/documentation.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
)

add_test(
  NAME project_structure_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/project_structure.py"
)

add_test(
  NAME verification_schema_contracts
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_verification_contracts.py"
)

add_test(
  NAME canonical_oracle_cases
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/cases.py"
    --probe "$<TARGET_FILE:ledger_probe>"
    --case-dir "${CMAKE_SOURCE_DIR}/verification/cases"
    --artifact-dir "${HBFSIM_TEST_OUT}/foundational-validation"
)

add_test(
  NAME oracle_property_fuzz
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/fuzz.py"
    --probe "$<TARGET_FILE:ledger_probe>"
    --seeds 32
    --max-requests 50
    --artifact-dir "${HBFSIM_TEST_OUT}/foundational-property-fuzz"
)

add_test(
  NAME external_evidence_contracts
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_external_evidence.py"
)

add_test(
  NAME placement_oracle_differential
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/placement.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --config
      "${CMAKE_SOURCE_DIR}/configs/systems/server-hbm128-hbf512.cfg"
    --seeds 12
    --operations 28
    --report
      "${HBFSIM_TEST_OUT}/foundational-behavioral-differential.json"
)

add_test(
  NAME analytical_microbench
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/analytical.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --config
      "${CMAKE_SOURCE_DIR}/configs/systems/eight-stack-baseline.cfg"
    --repository "${CMAKE_SOURCE_DIR}"
    --artifact-dir
      "${HBFSIM_TEST_OUT}/foundational-analytical-microbench"
    --report
      "${HBFSIM_TEST_OUT}/foundational-analytical-microbench.json"
)

add_test(
  NAME certificate_contracts
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_verification_certificate.py"
)

add_test(
  NAME system_config_contracts
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_system_configs.py"
)

add_test(
  NAME hbf_ocp_v070_grades_effective
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/verification/gates/hbf_public_spec.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --config-root "${CMAKE_SOURCE_DIR}/configs"
    --registry "${CMAKE_SOURCE_DIR}/configs/parameter-provenance.json"
    --output-dir "${HBFSIM_TEST_OUT}/hbf-ocp-v070-grades"
)
