#ifndef _DG_TYPEDEFS_CUH_
#define _DG_TYPEDEFS_CUH_
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "sparseblockmat.h"
#include "sparsematrix.h"

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
using cHVec = thrust::host_vector<thrust::complex<double>>; //!< complex Host Vector
using iHVec = thrust::host_vector<int>; //!< integer Host Vector
using fHVec = thrust::host_vector<float>; //!< Host Vector

using DVec  = thrust::device_vector<double>; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu or THRUST_DEVICE_SYSTEM_CPP for a cpu.
using iDVec = thrust::device_vector<int>; //!< integer Device Vector
using cDVec = thrust::device_vector<thrust::complex<double>>; //!< complex Device Vector
using fDVec = thrust::device_vector<float>; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu or THRUST_DEVICE_SYSTEM_CPP for a cpu.

//derivative matrices
template<class T>
using HMatrix_t = EllSparseBlockMat<T, thrust::host_vector>;
using HMatrix = EllSparseBlockMat<double, thrust::host_vector>; //!< Host Matrix for derivatives
using fHMatrix = EllSparseBlockMat<float, thrust::host_vector>; //!< Host Matrix for derivatives
using DMatrix = EllSparseBlockMat<double, thrust::device_vector>; //!< Device Matrix for derivatives
using fDMatrix = EllSparseBlockMat<float, thrust::device_vector>; //!< Device Matrix for derivatives

// Interpolation matrices
template<class real_type>
using IHMatrix_t = dg::SparseMatrix<int, real_type, thrust::host_vector>;
template<class real_type>
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
//Ell matrix can be almost 3x faster than csr for GPU
//However, sometimes matrices contain outlier rows that do not fit in ell
using IDMatrix_t = dg::SparseMatrix<int, real_type, thrust::device_vector>;
#else
// csr matrix can be much faster than ell for CPU (we have our own symv implementation!)
using IDMatrix_t = dg::SparseMatrix<int, real_type, thrust::device_vector>;
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
using MHVec_t   = dg::MPI_Vector<dg::HVec_t<T> >; //!< MPI Host Vector s.a. dg::HVec_t
using MHVec     = dg::MPI_Vector<dg::HVec >; //!< MPI Host Vector s.a. dg::HVec
using cMHVec    = dg::MPI_Vector<dg::cHVec >; //!< MPI Host Vector s.a. dg::cHVec
using fMHVec    = dg::MPI_Vector<dg::fHVec >; //!< MPI Host Vector s.a. dg::fHVec
using iMHVec    = dg::MPI_Vector<dg::iHVec >; //!< MPI Host Vector s.a. dg::iHVec
using MDVec     = dg::MPI_Vector<dg::DVec >; //!< MPI Device Vector s.a. dg::DVec
using cMDVec    = dg::MPI_Vector<dg::cDVec >; //!< MPI Device Vector s.a. dg::cDVec
using fMDVec    = dg::MPI_Vector<dg::fDVec >; //!< MPI Device Vector s.a. dg::fDVec
using iMDVec    = dg::MPI_Vector<dg::iDVec >; //!< MPI Device Vector s.a. dg::iDVec

// TODO These may be private
template< class T>
using CooMat_t  = dg::CooSparseBlockMat<T, thrust::host_vector>;
using CooMat    = dg::CooSparseBlockMat<double, thrust::host_vector>;
using fCooMat   = dg::CooSparseBlockMat<float, thrust::host_vector>;
using DCooMat   = dg::CooSparseBlockMat<double, thrust::device_vector>;
using fDCooMat  = dg::CooSparseBlockMat<float, thrust::device_vector>;

template<class T>
using MHMatrix_t  = dg::MPISparseBlockMat<thrust::host_vector, dg::HMatrix_t<T>, dg::CooMat_t<T>>; //!< MPI Host Matrix for derivatives
using MHMatrix    = dg::MPISparseBlockMat<thrust::host_vector, dg::HMatrix, dg::CooMat>; //!< MPI Host Matrix for derivatives
using fMHMatrix   = dg::MPISparseBlockMat<thrust::host_vector, dg::fHMatrix, dg::fCooMat>; //!< MPI Host Matrix for derivatives
using MDMatrix    = dg::MPISparseBlockMat<thrust::device_vector, dg::DMatrix, dg::DCooMat>; //!< MPI Device Matrix for derivatives
using fMDMatrix   = dg::MPISparseBlockMat<thrust::device_vector, dg::fDMatrix, dg::fDCooMat>; //!< MPI Device Matrix for derivatives

template<class real_type>
using MIHMatrix_t = MPIDistMat< thrust::host_vector, IHMatrix_t<real_type> >;
template<class real_type>
using MIDMatrix_t = MPIDistMat< thrust::device_vector, IDMatrix_t<real_type> >;
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
using cHVec = cMHVec;
using fHVec = fMHVec;
using iHVec = iMHVec;

using DVec  = MDVec;
using cDVec = cMDVec;
using fDVec = fMDVec;
using iDVec = iMDVec;

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
using cHVec = cHVec;
using fHVec = fHVec;
using iHVec = iHVec;

using DVec  = DVec;
using cDVec = cDVec;
using fDVec = fDVec;
using iDVec = iDVec;

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
