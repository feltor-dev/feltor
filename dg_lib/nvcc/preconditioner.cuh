#ifndef _DG_PRECONDITIONER_
#define _DG_PRECONDITIONER_

namespace dg{

template< class Derived>
struct DiagonalPreconditioner
{
    __host__ __device__
    double operator()( int i) const {
        return static_cast<Derived*>(this)->implementation( i);
};


/**
* @brief The Preconditioner T 
*
* @ingroup containers
* T is the inverse of S 
* @tparam n Number of Legendre nodes per cell.
*/
template< size_t n>
struct T : public DiagonalPreconditioner< T<n> > 
{
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ T( double h = 2.):h_(h){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const double& h() const {return h_;}
    __host__ __device__ double implementation( int i) const {
        return (double)(2*(i%n)+1)/h_;
    }
  private:
    double h_;
};


/**
* @brief The Preconditioner S 
*
* @ingroup containers
* Elements of S are the scalar products of Legendre functions.
* Use in Scalar product.
* @tparam n Number of Legendre nodes per cell.
*/
template< size_t n>
struct S : public DiagonalPreconditioner < S <n> >
{
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ S( double h = 2.):h_(h){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const double& h() const {return h_;}
    __host__ __device__ double implementation( int i) const {
        return h_/(double)(2*(i%n)+1);
  private:
    double h_;
};


} //namespace dg
#include "blas/preconditioner.cuh"

#endif //_DG_PRECONDITIONER_
