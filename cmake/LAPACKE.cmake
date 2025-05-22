# Just an instruction to download the actual FindLAPACKE.cmake
# We store it in the CPM_SOURCE_CACHE and add the location to CMAKE_MODULE_PATH
set(LAPACKE_HASH_SUM "25732a8161212f69daa79350d1407fc5a7af61fab71ffc21d7750dc6624160d6")

if(CPM_SOURCE_CACHE)
  set(LAPACKE_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/lapacke")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(LAPACKE_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/lapacke")
else()
  set(LAPACKE_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake")
endif()

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(LAPACKE_DOWNLOAD_LOCATION ${LAPACKE_DOWNLOAD_LOCATION} ABSOLUTE)

file(DOWNLOAD
    https://github.com/isl-org/Open3D/raw/refs/heads/main/3rdparty/cmake/FindLAPACKE.cmake
    "${LAPACKE_DOWNLOAD_LOCATION}/FindLAPACKE.cmake" EXPECTED_HASH SHA256=${LAPACKE_HASH_SUM}
)

list(APPEND CMAKE_MODULE_PATH "${LAPACKE_DOWNLOAD_LOCATION}")
message( "Module path is ${CMAKE_MODULE_PATH}")
