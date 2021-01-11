#ifndef _DG_TYPEDEFS_CUH_
#define _DG_TYPEDEFS_CUH_
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "sparseblockmat.h"
#include "sparseblockmat.cuh"

/*! @file
  @brief Useful typedefs of commonly used types.
  */

namespace dg{

///@addtogroup typedefs
///@{
//vectors
using HVec  = thrust::host_vector<double>; //!< Host Vector
using iHVec = thrust::host_vector<int>; //!< integer Host Vector
using fHVec = thrust::host_vector<float>; //!< Host Vector

using DVec  = thrust::device_vector<double>; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu.
using iDVec = thrust::device_vector<int>; //!< integer Device Vector
using fDVec = thrust::device_vector<float>; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu.

//derivative matrices
using HMatrix = EllSparseBlockMat<double>; //!< Host Matrix for derivatives
using fHMatrix = EllSparseBlockMat<float>; //!< Host Matrix for derivatives
using DMatrix = EllSparseBlockMatDevice<double>; //!< Device Matrix for derivatives
using fDMatrix = EllSparseBlockMatDevice<float>; //!< Device Matrix for derivatives
///@}
}//namespace dg

#ifdef MPI_VERSION
#include "mpi_vector.h"
#include "mpi_matrix.h"

namespace dg{
///@addtogroup typedefs
///@{
//using MPI_Vector<thrust::device_vector<double> >  MDVec; //!< MPI Device Vector s.a. dg::DVec
using MHVec     = dg::MPI_Vector<dg::HVec >; //!< MPI Host Vector s.a. dg::HVec
using fMHVec    = dg::MPI_Vector<dg::fHVec >; //!< MPI Host Vector s.a. dg::fHVec
using MDVec     = dg::MPI_Vector<dg::DVec >; //!< MPI Device Vector s.a. dg::DVec
using fMDVec    = dg::MPI_Vector<dg::fDVec >; //!< MPI Device Vector s.a. dg::fDVec

template<class real_type>
using NNCH = dg::NearestNeighborComm<dg::iHVec, thrust::host_vector<const real_type*>, thrust::host_vector<real_type> >; //!< host Communicator for the use in an mpi matrix for derivatives
template<class real_type>
using NNCD = dg::NearestNeighborComm<dg::iDVec, thrust::device_vector<const real_type*>, thrust::device_vector<real_type> >; //!< host Communicator for the use in an mpi matrix for derivatives
using dNNCH = dg::NNCH<double>; //!< host Communicator for the use in an mpi matrix for derivatives
using fNNCH = dg::NNCH<float>; //!< host Communicator for the use in an mpi matrix for derivatives
using dNNCD = dg::NNCD<double>; //!< device Communicator for the use in an mpi matrix for derivatives
using fNNCD = dg::NNCD<float>; //!< device Communicator for the use in an mpi matrix for derivatives

using CooMat    = dg::CooSparseBlockMat<double>;
using fCooMat   = dg::CooSparseBlockMat<float>;
using DCooMat   = dg::CooSparseBlockMatDevice<double>;
using fDCooMat  = dg::CooSparseBlockMatDevice<float>;

using MHMatrix  = dg::RowColDistMat<dg::HMatrix, dg::CooMat, dg::dNNCH>; //!< MPI Host Matrix for derivatives
using fMHMatrix = dg::RowColDistMat<dg::fHMatrix, dg::fCooMat, dg::fNNCH>; //!< MPI Host Matrix for derivatives
using MDMatrix  = dg::RowColDistMat<dg::DMatrix, dg::DCooMat, dg::dNNCD>; //!< MPI Device Matrix for derivatives
using fMDMatrix = dg::RowColDistMat<dg::fDMatrix, dg::fDCooMat, dg::fNNCD>; //!< MPI Device Matrix for derivatives

///@}
}//namespace dg
#endif //MPI_VERSION

//MPI-independent definitions
namespace dg{
///@addtogroup typedefs
///@{
//vectors
#ifdef MPI_VERSION
using HVec_t  = MHVec;
using fHVec_t = fMHVec;

using DVec_t  = MDVec;
using fDVec_t = fMDVec;

//derivative matrices
using HMatrix_t = MHMatrix;
using fHMatrix_t = fMHMatrix;
using DMatrix_t = MDMatrix;
using fDMatrix_t = fMDMatrix;
#else
using HVec_t  = HVec;
using fHVec_t = fHVec;

using DVec_t  = DVec;
using fDVec_t = fDVec;

//derivative matrices
using HMatrix_t = HMatrix;
using fHMatrix_t = fHMatrix;
using DMatrix_t = DMatrix;
using fDMatrix_t = fDMatrix;
#endif //MPI_VERSION
///@}
}//namespace dg

#endif//_DG_TYPEDEFS_CUH_
