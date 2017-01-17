#ifndef _DG_TYPEDEFS_CUH_
#define _DG_TYPEDEFS_CUH_


#define FELTOR_MAJOR_VERSION 3
#define FELTOR_MINOR_VERSION 2
#define FELTOR_SUBMINOR_VERSION 0

/*! @file

  This file contains useful typedefs of commonly used types.
  */
namespace dg{

///@addtogroup typedefs
///@{
//vectors
typedef cusp::array1d<double, cusp::device_memory> DVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
//typedef thrust::device_vector<double> DVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
typedef thrust::host_vector<double>   HVec; //!< Host Vector
//typedef thrust::device_vector<int> IDVec; //!< integer Device Vector
typedef cusp::array1d<int, cusp::device_memory> iDVec; //!< integer Device Vector
typedef thrust::host_vector<int>   iHVec; //!< integer Host Vector
//derivative matrices
typedef EllSparseBlockMatDevice<double> DMatrix; //!< Device Matrix for derivatives
typedef EllSparseBlockMat<double> HMatrix; //!< Host Matrix for derivatives

#ifdef MPI_VERSION
//typedef MPI_Vector<thrust::device_vector<double> >  MDVec; //!< MPI Device Vector s.a. dg::DVec
typedef MPI_Vector<DVec >  MDVec; //!< MPI Device Vector s.a. dg::DVec
typedef MPI_Vector<thrust::host_vector<double>  >   MHVec; //!< MPI Host Vector

typedef NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<double> > NNCH; //!< host Communicator for the use in an mpi matrix for derivatives
//typedef NearestNeighborComm<thrust::device_vector<int>, thrust::device_vector<double> > NNCD; //!< device Communicator for the use in an mpi matrix for derivatives
typedef NearestNeighborComm<iDVec, DVec > NNCD; //!< device Communicator for the use in an mpi matrix for derivatives

typedef dg::RowColDistMat<dg::EllSparseBlockMat<double>, dg::CooSparseBlockMat<double>, dg::NNCH> MHMatrix; //!< MPI Host Matrix for derivatives
typedef dg::RowColDistMat<dg::EllSparseBlockMatDevice<double>, dg::CooSparseBlockMatDevice<double>, dg::NNCD> MDMatrix; //!< MPI Device Matrix for derivatives
#endif
//////////////////////////////////////////////FLOAT VERSIONS////////////////////////////////////////////////////
//vectors
typedef cusp::array1d<float, cusp::device_memory> fDVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
//typedef thrust::device_vector<float> fDVec; //!< Device Vector. The device can be an OpenMP parallelized cpu or a gpu. This depends on the value of the macro THRUST_DEVICE_SYSTEM, which can be either THRUST_DEVICE_SYSTEM_OMP for openMP or THRUST_DEVICE_SYSTEM_CUDA for a gpu. 
typedef thrust::host_vector<float>   fHVec; //!< Host Vector
//derivative matrices
typedef EllSparseBlockMatDevice<float> fDMatrix; //!< Device Matrix for derivatives
typedef EllSparseBlockMat<float> fHMatrix; //!< Host Matrix for derivatives

#ifdef MPI_VERSION
typedef MPI_Vector<thrust::device_vector<float> >  fMDVec; //!< MPI Device Vector s.a. dg::DVec
typedef MPI_Vector<thrust::host_vector<float>  >   fMHVec; //!< MPI Host Vector

typedef NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<float> > fNNCH; //!< host Communicator for the use in an mpi matrix for derivatives
typedef NearestNeighborComm<thrust::device_vector<int>, thrust::device_vector<float> > fNNCD; //!< device Communicator for the use in an mpi matrix for derivatives

typedef dg::RowColDistMat<dg::EllSparseBlockMat<float>, dg::CooSparseBlockMat<float>, dg::fNNCH> fMHMatrix; //!< MPI Host Matrix for derivatives
typedef dg::RowColDistMat<dg::EllSparseBlockMatDevice<float>, dg::CooSparseBlockMatDevice<float>, dg::fNNCD> fMDMatrix; //!< MPI Device Matrix for derivatives
#endif
///@}

}//namespace dg

#endif//_DG_TYPEDEFS_CUH_
