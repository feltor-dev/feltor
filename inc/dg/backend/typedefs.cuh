#ifndef _DG_TYPEDEFS_CUH_
#define _DG_TYPEDEFS_CUH_

/*! @file

  This file contains useful typedefs of commonly used types.
  */
namespace dg{
//vectors
typedef thrust::device_vector<double> DVec; //!< Device Vector
typedef thrust::host_vector<double> HVec; //!< Host Vector
//derivative matrices
typedef EllSparseBlockMatDevice DMatrix;
typedef EllSparseBlockMat HMatrix;

#ifdef MPI_VERSION
typedef MPI_Vector<thrust::device_vector<double> > MDVec;
typedef MPI_Vector<thrust::host_vector<double>  >   MHVec;

typedef NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<double> > NNCH;
typedef NearestNeighborComm<thrust::device_vector<int>, thrust::device_vector<double> > NNCD;

typedef dg::RowColDistMat<dg::EllSparseBlockMat, dg::CooSparseBlockMat, dg::NNCH> MHMatrix;
typedef dg::RowColDistMat<dg::EllSparseBlockMatDevice, dg::CooSparseBlockMatDevice, dg::NNCD> MDMatrix;
#endif

}//namespace dg

#endif//_DG_TYPEDEFS_CUH_
