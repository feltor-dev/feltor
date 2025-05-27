# https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
include(CTest)
include(Catch)

### @brief Given a src file named test_path (read-only) and a runtime folder create a
### target with name target_name (write-only)
###
### @param test_path SRC file name to be compiled. Executable will be named after its STEM.
###     Target will be named "dg_testfolder_stem" or "dg_stem".
### @param test_folder Folder name of built executable (i.e. exe will live in
###     build/tests/${test_folder}/${STEM}). If empty string "", then exe will live
###     in build/tests/${STEM}.
### @param with_MPI If on, then executable is compiled with MPI
### @param target_name (write-only) Contains target name (dg_testfolder_stem or dg_stem) on output
###     This allows to link libraries on target_name in calling scope
function(add_dg_test test_path test_folder with_MPI target_name)
    if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
        set_source_files_properties("${test_path}" PROPERTIES LANGUAGE CUDA)
    endif()
    # Get a unique name for the test
    cmake_path(GET test_path STEM test_stem)
    if( test_folder STREQUAL "")
        set(test_alias "feltor::dg::${test_stem}")
        set(test_name "dg_${test_stem}")
    else()
        set(test_alias "feltor::dg::${test_folder}::${test_stem}")
        set(test_name "dg_${test_folder}_${test_stem}")
    endif()
    # Get custom main function
    if( with_MPI)
        set(test_main "${CMAKE_SOURCE_DIR}/tests/catch-tests-mpi.cpp")
    else()
        set(test_main "${CMAKE_SOURCE_DIR}/tests/catch-tests.cpp")
    endif()
    # Create the test executable and link dependencies
    add_executable(${test_name} "${test_main}" "${test_path}")
    add_executable(${test_alias} ALIAS ${test_name})
    # Change name of executable
    set_target_properties( ${test_name} PROPERTIES OUTPUT_NAME ${test_stem})
    # Create Alias name feltor::dg::...
    target_link_libraries(${test_name} PRIVATE dg_dg)
    target_link_libraries(${test_name} PRIVATE Catch2::Catch2)
    if( with_MPI)
        target_link_libraries(${test_name} PRIVATE MPI::MPI_CXX)
        target_compile_definitions(${test_name} PRIVATE WITH_MPI)
    endif()
    # Add as dependency to the dg_tests target
    add_dependencies(dg_tests ${test_name})
    # Build tests in ./build/tests
    if( test_folder STREQUAL "")
        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        )
    else()
        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/${test_folder}"
        )
    endif()
    # Register the tests with CTest
    catch_discover_tests(${test_name})
    # Set function output
    set( target_name ${test_name} PARENT_SCOPE)
endfunction()
