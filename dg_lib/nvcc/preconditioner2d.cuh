#ifndef _DG_PRECONDITIONER2D_
#define _DG_PRECONDITIONER2D_

namespace dg{

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
template< class T, size_t n>
struct MatrixTraits< T2D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

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
template< class T, size_t n>
struct MatrixTraits< S2D< T, n> > 
{
    typedef T value_type;
    typedef DiagonalPreconditionerTag matrix_category;
};

} //namespace dg

#endif //_DG_PRECONDITIONER2D_
