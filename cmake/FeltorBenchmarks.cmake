function(add_dg_benchmark benchmark_path)
  # Get a unique name for the benchmark
  cmake_path(GET benchmark_path STEM benchmark_stem)
  set(benchmark_name "dg_${benchmark_stem}")
  # Create the benchmark executable and link dependencies
  add_executable(${benchmark_name} "${benchmark_path}")
  target_link_libraries(${benchmark_name} PRIVATE feltor::dg)
  # Add as dependency to the dg_benchmarks target
  add_dependencies(dg_benchmarks ${benchmark_name})
  # Ensure benchmarks are built in their own directory
  set_target_properties(${benchmark_name} PROPERTIES 
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
  )
endfunction()
