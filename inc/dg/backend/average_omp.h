#pragma once

#include "exblas/exdot.fpe.cpp"
#include "vector_categories.h"
#ifdef MPI_VERSION
#include "exblas/mpi_accumulate.h"
#endif //MPI_VERSION

namespace dg
{
void transpose_dispatch( OmpTag, unsigned nx, unsigned ny, const double* RESTRICT in, double* RESTRICT out)
{
#pragma omp parallel for
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[j*ny+i] = in[i*nx+j];
}
template<class value_type>
void transpose_dispatch( OmpTag, unsigned nx, unsigned ny, const value_type* RESTRICT in, value_type* RESTRICT out)
{
#pragma omp parallel for
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[j*ny+i] = in[i*nx+j];
}
template<class value_type>
void extend_line( OmpTag, unsigned nx, unsigned ny, const value_type* RESTRICT in, value_type* RESTRICT out)
{
#pragma omp parallel for
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[i*nx+j] = in[j];
}
template<class value_type>
void extend_column( OmpTag, unsigned nx, unsigned ny, const value_type* RESTRICT in, value_type* RESTRICT out)
{
#pragma omp parallel for
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[i*nx+j] = in[i];
}

void average( OmpTag, unsigned nx, unsigned ny, const double* in0, const double* in1, double* out)
{
    static thrust::host_vector<int64_t> h_accumulator;
    static thrust::host_vector<double> h_round;
    h_accumulator.resize( ny*exdot_BIN_COUNT;
    for( unsigned i=0; i<ny; i++)
        exblas::exdot_omp(nx, &in0[i*nx], &in1[i*nx], &h_accumulator[i*exdot::BIN_COUNT]);
    for( unsigned i=0; i<ny; i++)
        out[i] = exblas::cpu::Round( &h_accumulator[i*exdot::BIN_COUNT]);
}

#ifdef MPI_VERSION
//local data plus communication
void average_mpi( OmpTag, unsigned nx, unsigned ny, const double* in0, const double* in1, double* out, MPI_Comm comm, MPI_Comm, comm_mod, MPI_Comm comm_mod_reduce )
{
    static thrust::host_vector<int64_t> h_accumulator;
    static thrust::host_vector<int64_t> h_accumulator2;
    h_accumulator2.resize( ny*exdot_BIN_COUNT;
    for( unsigned i=0; i<ny; i++)
        exblas::exdot_omp(nx, &in0[i*nx], &in1[i*nx], &h_accumulator2[i*exdot::BIN_COUNT]);
    h_accumulator.resize( h_accumulator2.size());
    reduce_mpi_cpu( ny, &h_accumulator2[0], &h_accumulator[0], comm, comm_mod, comm_mod_reduce);
    for( unsigned i=0; i<ny; i++)
        out[i] = exblas::cpu::Round( &h_accumulator[i*exdot::BIN_COUNT]);
}
#endif //MPI_VERSION

}//namespace dg
