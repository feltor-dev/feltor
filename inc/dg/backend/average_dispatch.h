#pragma once

#include "average_cpu.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "average_gpu.cuh"
#else
#include "average_omp.h"
#endif

namespace dg{

/*!@brief Transpose vector

 * @copydoc hide_container
 * @param nx number of columns in input vector (size of contiguous chunks) /rows in output vector
 * @param ny number of rows in input vector /columns in output vector
 * @param in input
 * @param out output (may not alias in)
*/
template<class container>
void transpose( unsigned nx, unsigned ny, const container& in, container& out)
{
    assert(&in != &out);
    const get_value_type<container>* in_ptr = thrust::raw_pointer_cast( in.data());
    get_value_type<container>* out_ptr = thrust::raw_pointer_cast( out.data());
    return transpose_dispatch( get_execution_policy<container>(), nx, ny, in_ptr, out_ptr);
}

/*!@brief Copy a line  into rows of output vector

 * @copydoc hide_container
 * @param nx size of the input vector/ number of columns (size of contiguous chunks) in output vector
 * @param ny number of rows in output vector
 * @param in input (size nx)
 * @param out output (may not alias in)
*/
template<class container>
void extend_line( unsigned nx, unsigned ny, const container& in, container& out)
{
    assert(&in != &out);
    const get_value_type<container>* in_ptr = thrust::raw_pointer_cast( in.data());
    get_value_type<container>* out_ptr = thrust::raw_pointer_cast( out.data());
    return extend_line( get_execution_policy<container>(), nx, ny, in_ptr, out_ptr);
}
/*!@brief Copy a line  into columns of output vector

 * @copydoc hide_container
 * @param nx number of columns (size of contiguous chunks) in output vector
 * @param ny size of input vector, number of rows in output vector
 * @param in input (size ny)
 * @param out output (may not alias in)
*/
template<class container>
void extend_column( unsigned nx, unsigned ny, const container& in, container& out)
{
    assert(&in != &out);
    const get_value_type<container>* in_ptr = thrust::raw_pointer_cast( in.data());
    get_value_type<container>* out_ptr = thrust::raw_pointer_cast( out.data());
    return extend_column( get_execution_policy<container>(), nx, ny, in_ptr, out_ptr);
}

template<class container>
void average( unsigned nx, unsigned ny, const container& in0, const container& in1, container& out)
{
    static_assert( std::is_same<get_value_type<container>, double>::value, "We only support double precision dot products at the moment!");
    const double* in0_ptr = thrust::raw_pointer_cast( in0.data());
    const double* in1_ptr = thrust::raw_pointer_cast( in1.data());
          double* out_ptr = thrust::raw_pointer_cast( out.data());
    average( get_execution_policy<container>(), nx, ny, in0_ptr, in1_ptr, out_ptr);
}

#ifdef MPI_VERSION
template<class container>
void mpi_average( unsigned nx, unsigned ny, const container& in0, const container& in1, container& out, MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce)
{
    static_assert( std::is_same<get_value_type<container>, double>::value, "We only support double precision dot products at the moment!");
    const double* in0_ptr = thrust::raw_pointer_cast( in0.data());
    const double* in1_ptr = thrust::raw_pointer_cast( in1.data());
          double* out_ptr = thrust::raw_pointer_cast( out.data());
    average_mpi( get_execution_policy<container>(), nx, ny, in0_ptr, in1_ptr, out_ptr, comm, comm_mod, comm_mod_reduce);
}
#endif //MPI_VERSION

}//namespace dg
