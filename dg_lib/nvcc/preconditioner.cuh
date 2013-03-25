#ifndef _DG_PRECONDITIONER_
#define _DG_PRECONDITIONER_

#include "matrix_categories.h"
#include "matrix_traits.h"

namespace dg{

template< class T = double>
struct Identity
{
    typedef T value_type;
    typedef IdentityTag matrix_category;
};

template<class T>
struct MatrixTraits<Identity< T>  >
{
    typedef T value_type;
    typedef typename Identity<T>::matrix_category matrix_category;
};

/*
template< class T, class Derived>
struct DiagonalPreconditioner
{
    typedef DiagonalPreconditionerTag matrix_category;
    typedef T value_type;
    __host__ __device__
    value_type operator()( int i) const {
        return static_cast<Derived*>(this)->implementation( i);
    }
};
*/


/**
* @brief The Preconditioner T 
*
* @ingroup containers
* T is the inverse of S 
* @tparam n Number of Legendre nodes per cell.
*/
template< class T, size_t n>
struct T1D //: public DiagonalPreconditioner< T, T1D<T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ T1D( value_type h = 2.):h_(h){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& h() const {return h_;}
    __host__ __device__ value_type operator()( int i) const 
    {
        return (value_type)(2*(i%n)+1)/h_;
    }
  private:
    value_type h_;
};

template< class T, size_t n>
struct MatrixTraits< T1D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};


/**
* @brief The Preconditioner S 
*
* @ingroup containers
* Elements of S are the scalar products of Legendre functions.
* Use in Scalar product.
* @tparam n Number of Legendre nodes per cell.
*/
template< class T, size_t n>
struct S1D// : public DiagonalPreconditioner < T, S1D <T, n> >
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ S1D( value_type h = 2.):h_(h){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& h() const {return h_;}
    __host__ __device__ value_type operator()( int i) const 
    {
        return h_/(value_type)(2*(i%n)+1);
    }
  private:
    value_type h_;
};
template< class T, size_t n>
struct MatrixTraits< S1D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};


} //namespace dg

#include "preconditioner2d.cuh"


#endif //_DG_PRECONDITIONER_
