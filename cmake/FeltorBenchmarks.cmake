

### @brief Given a src file named benchmark_path (read-only) and a runtime folder create a
### target with name target_name (write-only)
###
### @param benchmark_path SRC file name to be compiled. Executable will be named after its STEM
###     Target will be named "dg_benchmarkfolder_stem" or "dg_stem".
### @param benchmark_folder Folder name of built executable (i.e. exe will live in
###     build/benchmarks/${benchmark_folder}/${STEM}). If empty string "", then exe will live
###     in build/benchmarks/${STEM}.
### @param with_MPI If on, then executable is compiled with MPI
### @param target_name (write-only) Contains target name (dg_benchmarkfolder_stem or dg_stem) on output
###     This allows to link libraries on target_name in calling scope
function(add_dg_benchmark benchmark_path benchmark_folder with_MPI target_name)
    if(CCCL_THRUST_DEVICE_SYSTEM STREQUAL "CUDA" OR CCCL_THRUST_DEVICE_SYSTEM STREQUAL "")
        set_source_files_properties("${benchmark_path}" PROPERTIES LANGUAGE CUDA)
    else() # Necessary for matrix benchmarks
        set_source_files_properties("${benchmark_path}" PROPERTIES LANGUAGE CXX)
    endif()
    # Get a unique name for the benchmark
    cmake_path(GET benchmark_path STEM benchmark_stem)
    if( benchmark_folder STREQUAL "")
        set(benchmark_alias "feltor::dg::${benchmark_stem}")
        set(benchmark_name "dg_${benchmark_stem}")
    else()
        set(benchmark_alias "feltor::dg::${benchmark_folder}::${benchmark_stem}")
        set(benchmark_name "dg_${benchmark_folder}_${benchmark_stem}")
    endif()
    # Create the benchmark executable and link dependencies
    add_executable(${benchmark_name} "${benchmark_main}" "${benchmark_path}")
    add_executable(${benchmark_alias} ALIAS ${benchmark_name})
    # Change name of executable
    set_target_properties( ${benchmark_name} PROPERTIES OUTPUT_NAME ${benchmark_stem})
    target_link_libraries(${benchmark_name} PRIVATE dg_dg)
    if( with_MPI)
        target_link_libraries(${benchmark_name} PRIVATE MPI::MPI_CXX)
        target_compile_definitions(${benchmark_name} PRIVATE WITH_MPI)
    endif()
    # Add as dependency to the dg_benchmarks target
    add_dependencies(dg_benchmarks ${benchmark_name})
    # Build benchmarks in ./build/benchmarks
    if( benchmark_folder STREQUAL "")
        set_target_properties(${benchmark_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
        )
    else()
        set_target_properties(${benchmark_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks/${benchmark_folder}"
        )
    endif()
    set( target_name ${benchmark_name} PARENT_SCOPE)
endfunction()
