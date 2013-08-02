#ifndef _DG_PRECONDITIONER_
#define _DG_PRECONDITIONER_

#include "matrix_categories.h"
#include "matrix_traits.h"
#include "grid.cuh"

/*! @file 
  
  Contains special diagonal matrices usable as preconditioners in CG and as 
  weight functions in a general blas1::dot product for Gauss-Legendre integration
  */


namespace dg{

///@addtogroup creation
///@{
/**
* @brief The Identity matrix
*
* @tparam T Value type
*/
template< class T = double>
struct Identity
{
    typedef T value_type;
    typedef IdentityTag matrix_category;
};

/**
* @brief The diaonal mass-matrix S 
*
* Elements of S are the scalar products of Legendre functions.
* Use in Scalar products for vectors in l-space.
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template< class T>
struct S1D
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ S1D( value_type h):h_(h){}
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ S1D( const Grid1d<T>& g):n_(g.n()), h_(g.h()){}
    //__host__ __device__ const value_type& h() const {return h_;}
    __host__ __device__ value_type operator()( int i) const 
    {
        return h_/(value_type)(2*(i%n_)+1);
    }
  private:
    unsigned n_;
    value_type h_;
};
/**
* @brief The inverse of S 
*
* T is the inverse of S 
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template< class T>
struct T1D 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ T1D( value_type h):h_(h){}
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ T1D( const Grid1d<T>& g):n(g.n()),h_(g.h()){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& h() const {return h_;}
    __host__ __device__ value_type operator()( int i) const 
    {
        return (value_type)(2.*(i%n)+1.)/h_;
    }
  private:
    unsigned n;
    value_type h_;
};



///@}
///@cond
template<class T>
struct MatrixTraits<Identity< T>  >
{
    typedef T value_type;
    typedef typename Identity<T>::matrix_category matrix_category;
};
template< class T>
struct MatrixTraits< S1D< T > >
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

template< class T>
struct MatrixTraits< T1D< T > >
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

///@endcond



} //namespace dg

#include "preconditioner2d.cuh"


#endif //_DG_PRECONDITIONER_
