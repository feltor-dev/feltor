#ifndef _DG_PRECONDITIONER_
#define _DG_PRECONDITIONER_

#include "matrix_categories.h"
#include "matrix_traits.h"
#include "grid.cuh"
#include "dlt.h"

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
template< class T, size_t n>
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
    __host__ S1D( const Grid1d<T,n>& g):h_(g.h()){}
    //__host__ __device__ const value_type& h() const {return h_;}
    __host__ __device__ value_type operator()( int i) const 
    {
        return h_/(value_type)(2*(i%n)+1);
    }
  private:
    value_type h_;
};
/**
* @brief The inverse of S 
*
* T is the inverse of S 
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template< class T, size_t n>
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
    __host__ T1D( const Grid1d<T,n>& g):h_(g.h()){}
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
    value_type h_;
};


//W in x-space corresponds to S in l-space
/**
* @brief The diaonal weight-matrix W 
*
* Elements of W are the gaussian weights for Legendre functions.
* Use in Scalar products for vectors in x-space.
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template< class T, size_t n>
struct W1D
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ W1D( value_type h){ 
        for( unsigned i=0; i<n; i++)
            w[i] = h/2.*DLT<n>::weight[i];
    }
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ W1D( const Grid1d<T,n>& g){
        for( unsigned i=0; i<n; i++)
            w[i] = g.h()/2.*DLT<n>::weight[i];
    }
    __host__ __device__ value_type operator()( int i) const 
    {
        return (T)w[i%n]; 
    }
  private:
    double w[n];
};

/**
* @brief The inverse of W 
*
* V is the inverse of W 
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template< class T, size_t n>
struct V1D
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ V1D( value_type h){ 
        for( unsigned i=0; i<n; i++)
            x[i] = 2./h/DLT<n>::weight[i];
    }
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ V1D( const Grid1d<T,n>& g){
        for( unsigned i=0; i<n; i++)
            x[i] = 2./g.h()/DLT<n>::weight[i];
    }
    __host__ __device__ value_type operator()( int i) const 
    {
        return x[i%n]; 
    }
  private:
    double x[n]; //the more u store, the slower it becomes on gpu
};

///@}
///@cond
template<class T>
struct MatrixTraits<Identity< T>  >
{
    typedef T value_type;
    typedef typename Identity<T>::matrix_category matrix_category;
};
template< class T, size_t n>
struct MatrixTraits< S1D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

template< class T, size_t n>
struct MatrixTraits< T1D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

template< class T, size_t n>
struct MatrixTraits< W1D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

template< class T, size_t n>
struct MatrixTraits< V1D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};
///@endcond



} //namespace dg

#include "preconditioner2d.cuh"


#endif //_DG_PRECONDITIONER_
