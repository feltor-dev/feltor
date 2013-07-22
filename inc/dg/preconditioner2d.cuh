#ifndef _DG_PRECONDITIONER2D_
#define _DG_PRECONDITIONER2D_

#include "matrix_traits.h"
#include "grid.cuh"

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
template<typename T>
struct S2D
{
    typedef T value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    //__host__ __device__ 
    //S2D( value_type hx, value_type hy): hx_(hx), hy_(hy){}
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ S2D( const Grid<T>& g): n( g.n()), hx_(g.hx()), hy_( g.hy()){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& hx() const {return hx_;}
    __host__ __device__ const value_type& hy() const {return hy_;}
    __host__ __device__ value_type operator() ( int idx) const 
    {
        return  hx_*hy_/(value_type)((2*get_j( idx) + 1)*(2*get_i( idx) + 1));
    }
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    __host__ __device__ int get_i( int idx) const { return (idx%(n*n))/n;}
    //get j index from N again
    __host__ __device__ int get_j( int idx) const { return (idx%(n*n))%n;}
    unsigned n;
    value_type hx_, hy_;
};


/**
* @brief The inverse of S 
*
* T is the inverse of S 
* @tparam T value type
* @tparam n Number of Legendre nodes per cell.
*/
template< typename T>
struct T2D 
{
    typedef T value_type;
    /**
    * @brief Constructor
    *
    * @param h The grid size assumed to be constant.
    */
    //__host__ __device__ 
    //T2D( value_type hx, value_type hy):hx_(hx), hy_(hy){}
    /**
    * @brief Construct on grid
    *
    * @param g The grid
    */
    __host__ T2D( const Grid<T>& g):n(g.n()), hx_(g.hx()), hy_( g.hy()){}
    /**
    * @brief 
    *
    * @return The grid size
    */
    __host__ __device__ const value_type& hx() const {return hx_;}
    __host__ __device__ const value_type& hy() const {return hy_;}
    __host__ __device__ value_type operator() ( int idx) const 
    {
        return  (value_type)((2*get_j( idx) + 1)*(2*get_i( idx) + 1))/hy_/hx_;
    }
  private:
    //if N = k*n*n+i*n+j, then
    //get i index from N again
    __host__ __device__ int get_i( int idx)const  { return (idx%(n*n))/n;}
    //get j index from N again
    __host__ __device__ int get_j( int idx)const  { return (idx%(n*n))%n;}
    unsigned n; 
    value_type hx_, hy_;
};

///@}

///@cond
template< class T>
struct MatrixTraits< S2D< T> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

template< class T>
struct MatrixTraits< T2D< T> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};
///@endcond
} //namespace dg


#endif //_DG_PRECONDITIONER2D_
