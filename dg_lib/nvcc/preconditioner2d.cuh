#ifndef _DG_PRECONDITIONER2D_
#define _DG_PRECONDITIONER2D_

#include "matrix_traits.h"
#include "grid.cuh"
#include "dlt.h"

namespace dg{
///@addtogroup creation
///@{
/**
* @brief The diaonal mass-matrix S 
*
* Elements of S are the scalar products of Legendre functions.
* Use in Scalar products for vectors in l-space.
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template<typename T,  size_t n>
struct S2D
{
    typedef T value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ 
    S2D( value_type hx, value_type hy): hx_(hx), hy_(hy){}
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ S2D( const Grid<T,n>& g):hx_(g.hx()), hy_( g.hy()){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& hx() const {return hx_;}
    __host__ __device__ const value_type& hy() const {return hy_;}
    __host__ __device__ value_type operator() ( int idx) const 
    {
        return  hx_/(value_type)(2*get_j( idx) + 1)
               *hy_/(value_type)(2*get_i( idx) + 1);
    }
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    __host__ __device__ int get_i( int idx) const { return (idx%(n*n))/n;}
    //get j index from N again
    __host__ __device__ int get_j( int idx) const { return (idx%(n*n))%n;}
    value_type hx_, hy_;
};


/**
* @brief The inverse of S 
*
* T is the inverse of S 
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template< typename T, size_t n>
struct T2D 
{
    typedef T value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__ __device__ 
    T2D( value_type hx, value_type hy):hx_(hx), hy_(hy){}
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ T2D( const Grid<T,n>& g):hx_(g.hx()), hy_( g.hy()){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& hx() const {return hx_;}
    __host__ __device__ const value_type& hy() const {return hy_;}
    __host__ __device__ value_type operator() ( int idx) const 
    {
        return  (value_type)(2*get_j( idx) + 1)/hx_
               *(value_type)(2*get_i( idx) + 1)/hy_;
    }
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    __host__ __device__ int get_i( int idx)const  { return (idx%(n*n))/n;}
    //get j index from N again
    __host__ __device__ int get_j( int idx)const  { return (idx%(n*n))%n;}
    value_type hx_, hy_;
};

//W2D in x-space corresponds to S2D in l-space
//for W2D and V2D a thrust::vector might be faster
/**
* @brief The diaonal weight-matrix W 
*
* Elements of W are the gaussian weights for Legendre functions.
* Use in Scalar products for vectors in x-space.
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template<typename T,  size_t n>
struct W2D
{
    typedef T value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__  
    W2D( value_type hx, value_type hy): hx_(hx), hy_(hy){
        for( unsigned i=0; i<n; i++)
            w[i] = DLT<n>::weight[i];
    }
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ W2D( const Grid<T,n>& g):hx_(g.hx()), hy_( g.hy()){
        for( unsigned i=0; i<n; i++)
            w[i] = DLT<n>::weight[i];
    }
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& hx() const {return hx_;}
    __host__ __device__ const value_type& hy() const {return hy_;}
    __host__ __device__ value_type operator() ( int idx) const 
    {
        return  hx_/2.*(value_type)(w[get_j( idx)])
               *hy_/2.*(value_type)(w[get_i( idx)]);
    }
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    __host__ __device__ int get_i( int idx) const { return (idx%(n*n))/n;}
    //get j index from N again
    __host__ __device__ int get_j( int idx) const { return (idx%(n*n))%n;}
    double w[n];
    value_type hx_, hy_;
};
/**
* @brief The inverse of W 
*
* V is the inverse of W 
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template<typename T,  size_t n>
struct V2D
{
    typedef T value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    __host__  
    V2D( value_type hx, value_type hy): hx_(hx), hy_(hy){
        for( unsigned i=0; i<n; i++)
            x[i] = DLT<n>::weight[i];
    }
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ V2D( const Grid<T,n>& g):hx_(g.hx()), hy_( g.hy()){
        for( unsigned i=0; i<n; i++)
            x[i] = DLT<n>::weight[i];
    }
    __host__ __device__ const value_type& hx() const {return hx_;}
    __host__ __device__ const value_type& hy() const {return hy_;}
    __host__ __device__ value_type operator() ( int idx) const 
    {
        return  2./hx_/(value_type)(x[get_j( idx)])
               *2./hy_/(value_type)(x[get_i( idx)]);
    }
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    __host__ __device__ int get_i( int idx) const { return (idx%(n*n))/n;}
    //get j index from N again
    __host__ __device__ int get_j( int idx) const { return (idx%(n*n))%n;}
    double x[n];
    value_type hx_, hy_;
};

///@}

///@cond
template< class T, size_t n>
struct MatrixTraits< S2D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

template< class T, size_t n>
struct MatrixTraits< T2D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};
template< class T, size_t n>
struct MatrixTraits< W2D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};
template< class T, size_t n>
struct MatrixTraits< V2D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};
///@endcond
} //namespace dg


#endif //_DG_PRECONDITIONER2D_
