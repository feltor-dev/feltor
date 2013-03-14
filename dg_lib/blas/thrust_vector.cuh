#ifndef _DG_THRUST_BLAS_VECTOR_
#define _DG_THRUST_BLAS_VECTOR_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include "../blas.h"
#include "../array.cuh"
#include "../quadmat.cuh"

//specialize BLAS1 functions for several containers
namespace dg{

struct daxpby_functor
{
    daxpby_functor( double alpha, double beta): alpha(alpha), beta(beta) {}
    template<size_t n>
    __host__ __device__
        Array<double, n>& operator()( const Array<double,n >& x, Array<double, n>& y)
        {
            //daxpby( alpha, x, beta, y);
            for( unsigned i=0; i<n; i++)
                y(i) = alpha*x(i)+beta*y(i);
            return y;
        }
    __host__ __device__
        double operator()( const double& x, const double& y)
        {
            return alpha*x+beta*y;
        }
  private:
    double alpha, beta;
};

/*
namespace cast{
template< class T, class Iterator>
thrust::device_ptr<T> device_cast ( const Iterator& it){
    T* raw_ptr_x = reinterpret_cast<T*>( thrust::raw_pointer_cast(it));
    return thrust::device_pointer_cast( raw_ptr_x);
}
template< class T, class Iterator>
T* device_cast ( const Iterator& it){
    return reinterpret_cast<double const*>( thrust::raw_pointer_cast(it));
}


} //namespace cast
*/

template< size_t n> 
struct dg::BLAS1< thrust::device_vector< dg::Array< double, n> > >
{
    typedef thrust::device_vector<dg::Array<double, n> > Vector;
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        //get a raw pointer and use thrust::transform
        //this is faster than directly using transform on Array
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        thrust::device_ptr<double const> d_x = thrust::device_pointer_cast( raw_ptr_x);
        double * raw_ptr_y = reinterpret_cast<double *>( thrust::raw_pointer_cast(y.data()));
        thrust::device_ptr<double> d_y = thrust::device_pointer_cast( raw_ptr_y);
        thrust::transform( d_x, d_x+n*N, d_y, d_y, daxpby_functor( alpha, beta));
    }
    static double ddot( const Vector& x, const Vector& y)
    {
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        thrust::device_ptr<double const> d_x = thrust::device_pointer_cast( raw_ptr_x);
        double const* raw_ptr_y = reinterpret_cast<double const*>( thrust::raw_pointer_cast(y.data()));
        thrust::device_ptr<double const> d_y = thrust::device_pointer_cast( raw_ptr_y);
        return thrust::inner_product( d_x, d_x+n*N, d_y, 0.0);
    }

};

template< size_t n> 
struct dg::BLAS1< thrust::host_vector< dg::Array< double, n> > >
{
    typedef thrust::host_vector< dg::Array< double, n> > Vector;
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        double const* raw_ptr_y = reinterpret_cast<double const*>( thrust::raw_pointer_cast(y.data()));
        thrust::transform( raw_ptr_x, raw_ptr_x+n*N, raw_ptr_y, raw_ptr_y, daxpby_functor( alpha, beta));
    }
    static double ddot( const Vector& x, const Vector& y)
    {
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        double const* raw_ptr_y = reinterpret_cast<double const*>( thrust::raw_pointer_cast(y.data()));
        return thrust::inner_product( raw_ptr_x, raw_ptr_x+n*N, raw_ptr_y, 0.0);
    }
};

template< size_t n> 
struct dg::BLAS1< thrust::host_vector< dg::QuadMat< double, n> > >
{
    typedef thrust::host_vector< dg::QuadMat< double, n> > Vector;
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        double * raw_ptr_y = reinterpret_cast<double *>( thrust::raw_pointer_cast(y.data()));
        thrust::transform( raw_ptr_x, raw_ptr_x+n*n*N, raw_ptr_y, raw_ptr_y, daxpby_functor( alpha, beta));
    }
    static double ddot( const Vector& x, const Vector& y)
    {
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        double const * raw_ptr_y = reinterpret_cast<double const*>( thrust::raw_pointer_cast(y.data()));
        return thrust::inner_product( raw_ptr_x, raw_ptr_x+n*n*N, raw_ptr_y, 0.0);
    }
};
template< size_t n> 
struct dg::BLAS1< thrust::device_vector< dg::QuadMat< double, n> > >
{
    typedef thrust::device_vector< dg::QuadMat< double, n> > Vector;
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        thrust::device_ptr<double const> d_x = thrust::device_pointer_cast( raw_ptr_x);
        double * raw_ptr_y = reinterpret_cast<double *>( thrust::raw_pointer_cast(y.data()));
        thrust::device_ptr<double> d_y = thrust::device_pointer_cast( raw_ptr_y);
        thrust::transform( d_x, d_x+n*n*N, d_y, d_y, daxpby_functor( alpha, beta));
    }
    static double ddot( const Vector& x, const Vector& y)
    {
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        thrust::device_ptr<double const> d_x = thrust::device_pointer_cast( raw_ptr_x);
        double const * raw_ptr_y = reinterpret_cast<double const*>( thrust::raw_pointer_cast(y.data()));
        thrust::device_ptr<double const> d_y = thrust::device_pointer_cast( raw_ptr_y);
        return thrust::inner_product( d_x, d_x+n*n*N, d_y, 0.0);
    }
};


} //namespace dg


#endif //_DG_THRUST_BLAS_VECTOR_

