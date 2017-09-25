#pragma once

#include "projection.cuh"

namespace dg
{
///@addtogroup typedefs
///@{
//interpolation matrices
typedef MPIDistMat< cusp::csr_matrix<int, double, cusp::host_memory>, GeneralComm< thrust::host_vector<int>, thrust::host_vector<double> >   MIHMatrix; //!< MPI distributed CSR host Matrix
typedef MPIDistMat< cusp::csr_matrix<int, double, cusp::device_memory>, GeneralComm< thrust::host_vector<int>, thrust::host_vector<double> > MIDMatrix; //!< MPI distributed CSR device Matrix
///@}



}//namespace dg
