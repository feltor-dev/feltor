function(add_dg_test test_path)
  # Get a unique name for the test
  cmake_path(GET test_path STEM test_stem)
  set(test_name "dg_${test_stem}")
  # Optionally treat .cu files as .cpp files
  if(NOT FELTOR_USE_GPU)
      set_source_files_properties(${test_path} PROPERTIES LANGUAGE CXX)
  endif()
  # Create the test executable and link dependencies
  add_executable(${test_name} "${test_path}")
  target_link_libraries(${test_name} PRIVATE feltor::dg)
  # Add as dependency to the dg_tests target
  add_dependencies(dg_tests ${test_name})
  # Ensure built tests are built in their own directory
  set_target_properties(${test_name} PROPERTIES 
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )
  # Register the test with CTest
  add_test(NAME ${test_name} COMMAND "tests/${test_name}")
endfunction()