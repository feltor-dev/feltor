#ifndef _DG_PRECONDITIONER_
#define _DG_PRECONDITIONER_

namespace dg{

struct T{
    __host__ __device__ T( double h = 2.):h_(h){}
    __host__ __device__ const double& h() const {return h_;}
  private:
    double h_;
};

struct S{
    __host__ __device__ S( double h = 2.):h_(h){}
    __host__ __device__ const double& h() const {return h_;}
  private:
    double h_;
};

}
#include "blas/preconditioner.h"

#endif //_DG_PRECONDITIONER_
