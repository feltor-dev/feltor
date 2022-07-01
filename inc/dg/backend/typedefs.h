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
template<class T>
using HVec_t  = thrust::host_vector<T>; //!< Host Vector
using HVec  = thrust::host_vector<double>; //!< Host Vector
using iHVec = thrust::host_vector<int>; //!< integer Host Vector
using fHVec = thrust::host_vector<float>; //!< Host Vector

using DVec  = thrust::device_vector<double>; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu or THRUST_DEVICE_SYSTEM_CPP for a cpu.
using iDVec = thrust::device_vector<int>; //!< integer Device Vector
using fDVec = thrust::device_vector<float>; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu or THRUST_DEVICE_SYSTEM_CPP for a cpu.

//derivative matrices
template<class T>
using HMatrix_t = EllSparseBlockMat<T>;
using HMatrix = EllSparseBlockMat<double>; //!< Host Matrix for derivatives
using fHMatrix = EllSparseBlockMat<float>; //!< Host Matrix for derivatives
using DMatrix = EllSparseBlockMatDevice<double>; //!< Device Matrix for derivatives
using fDMatrix = EllSparseBlockMatDevice<float>; //!< Device Matrix for derivatives

// Interpolation matrices
template<class real_type>
using IHMatrix_t = cusp::csr_matrix<int, real_type, cusp::host_memory>;
template<class real_type>
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
//Ell matrix can be almost 3x faster than csr for GPU
//However, sometimes matrices contain outlier rows that do not fit in ell
using IDMatrix_t = cusp::csr_matrix<int, real_type, cusp::device_memory>;
#else
// csr matrix can be much faster than ell for CPU (we have our own symv implementation!)
using IDMatrix_t = cusp::csr_matrix<int, real_type, cusp::device_memory>;
#endif
using IHMatrix = IHMatrix_t<double>;
using IDMatrix = IDMatrix_t<double>;

///@}
}//namespace dg

#ifdef MPI_VERSION
#include "mpi_vector.h"
#include "mpi_matrix.h"

namespace dg{
///@addtogroup typedefs
///@{
//using MPI_Vector<thrust::device_vector<double> >  MDVec; //!< MPI Device Vector s.a. dg::DVec
template<class T>
using MHVec_t     = dg::MPI_Vector<dg::HVec_t<T> >; //!< MPI Host Vector s.a. dg::HVec_t
using MHVec     = dg::MPI_Vector<dg::HVec >; //!< MPI Host Vector s.a. dg::HVec
using fMHVec    = dg::MPI_Vector<dg::fHVec >; //!< MPI Host Vector s.a. dg::fHVec
using MDVec     = dg::MPI_Vector<dg::DVec >; //!< MPI Device Vector s.a. dg::DVec
using fMDVec    = dg::MPI_Vector<dg::fDVec >; //!< MPI Device Vector s.a. dg::fDVec

template<class T>
using NNCH = dg::NearestNeighborComm<dg::iHVec, thrust::host_vector<const T*>, thrust::host_vector<T> >; //!< host Communicator for the use in an mpi matrix for derivatives
template<class T>
using NNCD = dg::NearestNeighborComm<dg::iDVec, thrust::device_vector<const T*>, thrust::device_vector<T> >; //!< host Communicator for the use in an mpi matrix for derivatives
using dNNCH = dg::NNCH<double>; //!< host Communicator for the use in an mpi matrix for derivatives
using fNNCH = dg::NNCH<float>; //!< host Communicator for the use in an mpi matrix for derivatives
using dNNCD = dg::NNCD<double>; //!< device Communicator for the use in an mpi matrix for derivatives
using fNNCD = dg::NNCD<float>; //!< device Communicator for the use in an mpi matrix for derivatives

template< class T>
using CooMat_t    = dg::CooSparseBlockMat<T>;
using CooMat    = dg::CooSparseBlockMat<double>;
using fCooMat   = dg::CooSparseBlockMat<float>;
using DCooMat   = dg::CooSparseBlockMatDevice<double>;
using fDCooMat  = dg::CooSparseBlockMatDevice<float>;

template<class T>
using MHMatrix_t  = dg::RowColDistMat<dg::HMatrix_t<T>, dg::CooMat_t<T>, dg::NNCH<T>>; //!< MPI Host Matrix for derivatives
using MHMatrix  = dg::RowColDistMat<dg::HMatrix, dg::CooMat, dg::dNNCH>; //!< MPI Host Matrix for derivatives
using fMHMatrix = dg::RowColDistMat<dg::fHMatrix, dg::fCooMat, dg::fNNCH>; //!< MPI Host Matrix for derivatives
using MDMatrix  = dg::RowColDistMat<dg::DMatrix, dg::DCooMat, dg::dNNCD>; //!< MPI Device Matrix for derivatives
using fMDMatrix = dg::RowColDistMat<dg::fDMatrix, dg::fDCooMat, dg::fNNCD>; //!< MPI Device Matrix for derivatives

template<class real_type>
using MIHMatrix_t = MPIDistMat< IHMatrix_t<real_type>, GeneralComm< dg::iHVec, thrust::host_vector<real_type>> >;
template<class real_type>
using MIDMatrix_t = MPIDistMat< IDMatrix_t<real_type>, GeneralComm< dg::iDVec, thrust::device_vector<real_type>> >;
using MIHMatrix = MIHMatrix_t<double>;
using MIDMatrix = MIDMatrix_t<double>;


///@}
}//namespace dg
#endif //MPI_VERSION

//MPI-independent definitions
namespace dg{
///@addtogroup typedefs
///@{
//vectors
namespace x{
#ifdef MPI_VERSION
using HVec  = MHVec;
using fHVec = fMHVec;

using DVec  = MDVec;
using fDVec = fMDVec;

//derivative matrices
using HMatrix = MHMatrix;
using fHMatrix = fMHMatrix;
using DMatrix = MDMatrix;
using fDMatrix = fMDMatrix;
//interpolation matrices
using IHMatrix = MIHMatrix;
using IDMatrix = MIDMatrix;
#else
using HVec  = HVec;
using fHVec = fHVec;

using DVec  = DVec;
using fDVec = fDVec;

//derivative matrices
using HMatrix = HMatrix;
using fHMatrix = fHMatrix;
using DMatrix = DMatrix;
using fDMatrix = fDMatrix;
//interpolation matrices
using IHMatrix = IHMatrix;
using IDMatrix = IDMatrix;
#endif //MPI_VERSION
}//namespace x
///@}
}//namespace dg

////CONVENIENCE MACRO////////
#ifdef MPI_VERSION
#define DG_RANK0 if(rank==0)
#else //MPI_VERSION
#define DG_RANK0
#endif //MPI_VERSION

#endif//_DG_TYPEDEFS_CUH_
