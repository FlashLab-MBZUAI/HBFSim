# Workload tools, summary observability, report renderers and the ServeLoop
# frontend contract. Everything here runs against the built simulator
# programs or pure Python; nothing needs a research checkout except the
# 'serveloop'-labelled frontend test (label and fixture applied centrally).

add_test(
  NAME workload_quality_contracts
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_workload_quality.py"
)

add_test(
  NAME synthetic_trace_generator
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_synthetic_workload.py"
)

add_test(
  NAME trace_locality_analyzer
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_trace_analysis.py"
)

add_test(
  NAME trace_span_census_scalability
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_trace_span_census.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
)

add_test(
  NAME serveloop_frontend_contracts
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_serveloop_frontend.py"
    --simulator "$<TARGET_FILE:hbfsim>"
)
set_tests_properties(serveloop_frontend_contracts PROPERTIES TIMEOUT 120)

add_test(
  NAME hbf_controller_dram_ratio_effective
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_hbf_controller_dram_ratio.py"
    --reference "$<TARGET_FILE:hbfsim_reference>"
)

add_test(
  NAME reference_heatmap_disable_fast_path
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_reference_heatmap_disable.py"
    --reference "$<TARGET_FILE:hbfsim_reference>"
)

add_test(
  NAME summary_hybrid_residency_observability
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_summary_layer_streaming_stats.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
)

add_test(
  NAME summary_behavioral_tiering_v18_observability
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_summary_behavioral_tiering.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
    --config
      "${CMAKE_SOURCE_DIR}/configs/systems/server-hbm128-hbf512.cfg"
)

add_test(
  NAME lazy_sequential_equivalence_and_option_scoping
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_lazy_sequential_equivalence.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
)

add_test(
  NAME summary_time_breakdown_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_time_breakdown.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
)

add_test(
  NAME time_breakdown_report_renderer
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_time_breakdown_report.py"
)

add_test(
  NAME time_breakdown_visualization_renderer
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_plot_time_breakdown.py"
)

add_test(
  NAME waf_verifier_mutation_contract
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_waf_gate.py"
)

add_test(
  NAME gc_waf_quick_accounting
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_gc_waf_quick.py"
    --simulator "$<TARGET_FILE:hbfsim_reference>"
)
set_tests_properties(gc_waf_quick_accounting PROPERTIES TIMEOUT 60)

add_test(
  NAME address_heatmap_accounting
  COMMAND address_heatmap_test
)

add_test(
  NAME address_heatmap_renderer
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_plot_address_heatmap.py"
)

add_test(
  NAME address_heatmap_cpp_python_interop
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_SOURCE_DIR}/tests/python/test_address_heatmap_interop.py"
    --emitter "$<TARGET_FILE:address_heatmap_test>"
)
