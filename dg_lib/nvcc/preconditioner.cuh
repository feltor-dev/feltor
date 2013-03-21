#ifndef _DG_PRECONDITIONER_
#define _DG_PRECONDITIONER_

#include "matrix_categories.h"
namespace dg{

template< class Derived>
struct DiagonalPreconditioner
{
    typedef DiagonalPreconditionerTag matrix_category;
    typedef Derived::value_type value_type;
    __host__ __device__
    value_type operator()( int i) const {
        return static_cast<Derived*>(this)->implementation( i);
};


/**
* @brief The Preconditioner T 
*
* @ingroup containers
* T is the inverse of S 
* @tparam n Number of Legendre nodes per cell.
*/
template< class value_type, size_t n>
struct T : public DiagonalPreconditioner< T<n> > 
{
    typedef value_type value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ T( value_type h = 2.):h_(h){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& h() const {return h_;}
    __host__ __device__ value_type implementation( int i) const {
        return (value_type)(2*(i%n)+1)/h_;
    }
  private:
    value_type h_;
};


/**
* @brief The Preconditioner S 
*
* @ingroup containers
* Elements of S are the scalar products of Legendre functions.
* Use in Scalar product.
* @tparam n Number of Legendre nodes per cell.
*/
template< class value_type, size_t n>
struct S : public DiagonalPreconditioner < S <n> >
{
    typedef value_type value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ S( value_type h = 2.):h_(h){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& h() const {return h_;}
    __host__ __device__ value_type implementation( int i) const {
        return h_/(value_type)(2*(i%n)+1);
  private:
    value_type h_;
};


} //namespace dg
#include "blas/preconditioner.cuh"

#endif //_DG_PRECONDITIONER_
