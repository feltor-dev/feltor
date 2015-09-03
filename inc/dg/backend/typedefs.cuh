#ifndef _DG_TYPEDEFS_CUH_
#define _DG_TYPEDEFS_CUH_

/*! @file

  This file contains useful typedefs of commonly used types.
  */
namespace dg{
//vectors
typedef thrust::device_vector<double> DVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
typedef thrust::host_vector<double>   HVec; //!< Host Vector
typedef thrust::device_vector<int> IDVec; //!< integer Device Vector
typedef thrust::host_vector<int>   IHVec; //!< integer Host Vector
//derivative matrices
typedef EllSparseBlockMatDevice DMatrix; //!< Device Matrix for derivatives
typedef EllSparseBlockMat HMatrix; //!< Host Matrix for derivatives

#ifdef MPI_VERSION
typedef MPI_Vector<thrust::device_vector<double> >  MDVec; //!< MPI Device Vector s.a. dg::DVec
typedef MPI_Vector<thrust::host_vector<double>  >   MHVec; //!< MPI Host Vector

typedef NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<double> > NNCH; //!< host Communicator for the use in an mpi matrix for derivatives
typedef NearestNeighborComm<thrust::device_vector<int>, thrust::device_vector<double> > NNCD; //!< device Communicator for the use in an mpi matrix for derivatives

typedef dg::RowColDistMat<dg::EllSparseBlockMat, dg::CooSparseBlockMat, dg::NNCH> MHMatrix; //!< MPI Host Vector for derivatives
typedef dg::RowColDistMat<dg::EllSparseBlockMatDevice, dg::CooSparseBlockMatDevice, dg::NNCD> MDMatrix; //!< MPI Device Vector for derivatives
#endif

}//namespace dg

#endif//_DG_TYPEDEFS_CUH_
