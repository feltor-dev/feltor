#ifndef _DG_PRECONDITIONER_
#define _DG_PRECONDITIONER_

namespace dg{

struct T{
    T( double h = 2.):h_(h){}
    const double& h() const {return h_;}
  private:
    double h_;
};

struct S{
    S( double h = 2.):h_(h){}
    const double& h() const {return h_;}
  private:
    double h_;
};

}
#include "blas/preconditioner.h"

#endif //_DG_PRECONDITIONER_
