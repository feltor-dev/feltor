#ifndef _DG_PRECONDITIONER2D_
#define _DG_PRECONDITIONER2D_

namespace dg{
template< size_t n>
struct T2d{
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ 
    T2d( double hx = 2., double hy = 2.):hx_(hx), hy_(hy){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const double& hx() const {return hx_;}
    __host__ __device__ const double& hy() const {return hy_;}
  private:
    double hx_, hy_;
};
template< size_t n>
struct S2d{
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ 
    S2d( double hx = 2., double hy = 2.): hx_(hx), hy_(hy){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const double& hx() const {return hx_;}
    __host__ __device__ const double& hy() const {return hy_;}
  private:
    double hx_, hy_;
};
}
#include "blas/preconditioner.cuh"

#endif //_DG_PRECONDITIONER2D_
