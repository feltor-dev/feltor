#ifndef _DG_PRECONDITIONER2D_
#define _DG_PRECONDITIONER2D_

#include "preconditioner.h"

namespace dg{

template< size_t n>
struct T2d : public DiagonalPreconditioner< T2d< n> >
{
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
    __host__ __device__ double operator() ( int idx) const {
        return  (double)(2*get_j( idx) + 1)/hx
               *(double)(2*get_i( idx) + 1)/hy;
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    int get_i( int idx) { return (idx%(n*n))/n;}
    //get j index from N again
    int get_j( int idx) { return (idx%(n*n))%n;}
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
    __host__ __device__ double operator() ( int idx) const 
    {
        return  (double)(2*get_j( idx) + 1)/hx
               *(double)(2*get_i( idx) + 1)/hy;
    }
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    int get_i( int idx) { return (idx%(n*n))/n;}
    //get j index from N again
    int get_j( int idx) { return (idx%(n*n))%n;}
    double hx_, hy_;
};
}
#include "blas/preconditioner2d.cuh"

#endif //_DG_PRECONDITIONER2D_
