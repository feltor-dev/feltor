include(CTest)
include(Catch)

function(add_dg_test test_path)
  if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
    set_source_files_properties("${test_path}" PROPERTIES LANGUAGE CUDA)
  endif()
  # Get a unique name for the test
  cmake_path(GET test_path STEM test_stem)
  set(test_name "dg_${test_stem}")
  # Get custom main function
  set(test_main "${CMAKE_SOURCE_DIR}/tests/catch-tests.cpp")
  # Create the test executable and link dependencies
  add_executable(${test_name} "${test_main}" "${test_path}")
  target_link_libraries(${test_name} PRIVATE dg::dg)
  target_link_libraries(${test_name} PRIVATE Catch2::Catch2)
  # Add as dependency to the dg_tests target
  add_dependencies(dg_tests ${test_name})
  # Build tests in ./build/tests
  set_target_properties(${test_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )
  # Register the tests with CTest
  catch_discover_tests(${test_name})
endfunction()

function(add_dg_mpi_test test_path)
  if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
    set_source_files_properties("${test_path}" PROPERTIES LANGUAGE CUDA)
  endif()
  # Get a unique name for the test
  cmake_path(GET test_path STEM test_stem)
  set(test_name "dg_${test_stem}")
  # Get custom main function
  set(test_main "${CMAKE_SOURCE_DIR}/tests/catch-tests-mpi.cpp")
  # Create the test executable and link dependencies
  add_executable(${test_name} "${test_main}" "${test_path}")
  target_link_libraries(${test_name} PRIVATE dg::dg)
  target_link_libraries(${test_name} PRIVATE Catch2::Catch2)
  target_link_libraries(${test_name} PRIVATE MPI::MPI_CXX)
  target_compile_definitions(${test_name} PRIVATE WITH_MPI)
  # Add as dependency to the dg_tests target
  add_dependencies(dg_tests ${test_name})
  # Build tests in ./build/mpi_tests
  set_target_properties(${test_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/mpi_tests"
  )
  # Register the tests with CTest
  catch_discover_tests(${test_name})
endfunction()

function(add_file_test test_path)
  if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
    set_source_files_properties("${test_path}" PROPERTIES LANGUAGE CUDA)
  endif()
  # Get a unique name for the test
  cmake_path(GET test_path STEM test_stem)
  set(test_name "dg_file_${test_stem}")
  # Get custom main function
  set(test_main "${CMAKE_SOURCE_DIR}/tests/catch-tests.cpp")
  # Create the test executable and link dependencies
  add_executable(${test_name} "${test_main}" "${test_path}")
  target_link_libraries(${test_name} PRIVATE dg::dg)
  target_link_libraries(${test_name} PRIVATE dg::file)
  target_link_libraries(${test_name} PRIVATE Catch2::Catch2)
  # Add as dependency to the dg_tests target
  add_dependencies(dg_tests ${test_name})
  # Build tests in ./build/file_tests
  set_target_properties(${test_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/file_tests"
  )
  # Register the tests with CTest
  catch_discover_tests(${test_name})
endfunction()

function(add_geometries_test test_path)
  if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
    set_source_files_properties("${test_path}" PROPERTIES LANGUAGE CUDA)
  endif()
  # Get a unique name for the test
  cmake_path(GET test_path STEM test_stem)
  set(test_name "dg_geometries_${test_stem}")
  # Get custom main function
  set(test_main "${CMAKE_SOURCE_DIR}/tests/catch-tests.cpp")
  # Create the test executable and link dependencies
  add_executable(${test_name} "${test_main}" "${test_path}")
  target_link_libraries(${test_name} PRIVATE dg::dg)
  target_link_libraries(${test_name} PRIVATE dg::geometries)
  target_link_libraries(${test_name} PRIVATE Catch2::Catch2)
  # Add as dependency to the dg_tests target
  add_dependencies(dg_tests ${test_name})
  # Build tests in ./build/geometries_tests
  set_target_properties(${test_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/geometries_tests"
  )
  # Register the tests with CTest
  catch_discover_tests(${test_name})
endfunction()
