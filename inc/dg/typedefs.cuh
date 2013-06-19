#ifndef _DG_TYPEDEFS_CUH_
#define _DG_TYPEDEFS_CUH_

namespace dg{
typedef thrust::device_vector<double> DVec; //!< Device Vector
typedef thrust::host_vector<double> HVec; //!< Host Vector

typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix; //!< default matrix
typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix; //!< most efficient matrix format
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix; //!< most efficient matrix format

}//namespace dg

#endif//_DG_TYPEDEFS_CUH_
