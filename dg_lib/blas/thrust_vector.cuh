#ifndef _DG_THRUST_BLAS_VECTOR_
#define _DG_THRUST_BLAS_VECTOR_


#include "../blas.h"

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

template<>
template< class Array>
struct dg::BLAS1< thrust::device_vector< Array> >
{
    typedef thrust::device_vector<Array> Vector;
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        unsigned n = sizeof( Array)/8;
        std::cout << n <<"\n";
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        thrust::device_ptr<double const> d_x = thrust::device_pointer_cast( raw_ptr_x);
        double * raw_ptr_y = reinterpret_cast<double *>( thrust::raw_pointer_cast(y.data()));
        thrust::device_ptr<double> d_y = thrust::device_pointer_cast( raw_ptr_y);
        thrust::transform( d_x, d_x+n*N, d_y, d_y, daxpby_functor( alpha, beta));
    }
};

template<>
template< class Array>
struct dg::BLAS1< thrust::host_vector< Array> >
{
    typedef thrust::host_vector< Array> Vector;
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        unsigned n = sizeof( Array)/8;
        unsigned N = thrust::distance( x.begin(), x.end());
        double const * raw_ptr_x = reinterpret_cast<double const*>( thrust::raw_pointer_cast(x.data()));
        double * raw_ptr_y = reinterpret_cast<double *>( thrust::raw_pointer_cast(y.data()));
        thrust::transform( raw_ptr_x, raw_ptr_x+n*N, raw_ptr_y, raw_ptr_y, daxpby_functor( alpha, beta));
    }
};


} //namespace dg


#endif //_DG_THRUST_BLAS_VECTOR_

