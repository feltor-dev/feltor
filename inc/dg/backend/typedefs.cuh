#ifndef _DG_TYPEDEFS_CUH_
#define _DG_TYPEDEFS_CUH_

/*! @file
  @brief Useful typedefs of commonly used types.
  */

namespace dg{

///@addtogroup typedefs
///@{
//vectors
//typedef cusp::array1d<double, cusp::device_memory> DVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
typedef thrust::device_vector<double> DVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
typedef thrust::host_vector<double>   HVec; //!< Host Vector
typedef thrust::device_vector<int> iDVec; //!< integer Device Vector
//typedef cusp::array1d<int, cusp::device_memory> iDVec; //!< integer Device Vector
typedef thrust::host_vector<int>   iHVec; //!< integer Host Vector
//derivative matrices
typedef EllSparseBlockMatDevice<double> DMatrix; //!< Device Matrix for derivatives
typedef EllSparseBlockMat<double> HMatrix; //!< Host Matrix for derivatives

#ifdef MPI_VERSION
//typedef MPI_Vector<thrust::device_vector<double> >  MDVec; //!< MPI Device Vector s.a. dg::DVec
typedef MPI_Vector<dg::DVec >  MDVec; //!< MPI Device Vector s.a. dg::DVec
typedef MPI_Vector<dg::HVec >  MHVec; //!< MPI Host Vector s.a. dg::HVec

typedef NearestNeighborComm<dg::iHVec, dg::HVec > NNCH; //!< host Communicator for the use in an mpi matrix for derivatives
//typedef NearestNeighborComm<thrust::device_vector<int>, thrust::device_vector<double> > NNCD; //!< device Communicator for the use in an mpi matrix for derivatives
typedef NearestNeighborComm<dg::iDVec, dg::DVec > NNCD; //!< device Communicator for the use in an mpi matrix for derivatives

typedef dg::RowColDistMat<dg::HMatrix, dg::CooSparseBlockMat<double>, dg::NNCH> MHMatrix; //!< MPI Host Matrix for derivatives
typedef dg::RowColDistMat<dg::DMatrix, dg::CooSparseBlockMatDevice<double>, dg::NNCD> MDMatrix; //!< MPI Device Matrix for derivatives
#endif
//////////////////////////////////////////////FLOAT VERSIONS////////////////////////////////////////////////////
//vectors
//typedef cusp::array1d<float, cusp::device_memory> fDVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
typedef thrust::device_vector<float> fDVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
typedef thrust::host_vector<float>   fHVec; //!< Host Vector
//derivative matrices
typedef EllSparseBlockMatDevice<float> fDMatrix; //!< Device Matrix for derivatives
typedef EllSparseBlockMat<float> fHMatrix; //!< Host Matrix for derivatives

#ifdef MPI_VERSION
typedef MPI_Vector<dg::fDVec > fMDVec; //!< MPI Device Vector s.a. dg::DVec
typedef MPI_Vector<dg::fHVec > fMHVec; //!< MPI Host Vector

typedef NearestNeighborComm<dg::iHVec, dg::fHVec > fNNCH; //!< host Communicator for the use in an mpi matrix for derivatives
typedef NearestNeighborComm<dg::iDVec, dg::fDVec > fNNCD; //!< device Communicator for the use in an mpi matrix for derivatives

typedef dg::RowColDistMat<dg::fHMatrix, dg::CooSparseBlockMat<float>, dg::fNNCH> fMHMatrix; //!< MPI Host Matrix for derivatives
typedef dg::RowColDistMat<dg::fDMatrix, dg::CooSparseBlockMatDevice<float>, dg::fNNCD> fMDMatrix; //!< MPI Device Matrix for derivatives
#endif
///@}

}//namespace dg

#endif//_DG_TYPEDEFS_CUH_
