#ifndef _DG_BLAS_PRECONDITIONER2D_
#define _DG_BLAS_PRECONDITIONER2D_

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "../preconditioner2d.cuh"
#include "thrust_vector.cuh" //load thrust_vector BLAS1 routines

//namespace dg{
//
//namespace detail{
////if N = k*n*n+i*n+j, then
//
////get i index from N again
//template< size_t n>
//__host__ __device__ inline int get_i( int index)
//{
//    return (index%(n*n))/n;
//}
////get j index from N again
//template< size_t n>
//__host__ __device__ inline int get_j( int index)
//{
//    return (index%(n*n))%n;
//}
//template< size_t n>
//struct dsymv_functor_T2d
//{
//    typedef thrust::tuple< double, int> Pair;
//    dsymv_functor_T2d( double alpha, double beta): alpha(alpha), beta(beta) {}
//    __host__ __device__
//        double operator()(const double& x,  const Pair& p)
//        {
//            double y = alpha*x *(2.* (thrust::get<1>(p)%n)+1.)
//                        + beta*thrust::get<0>(p);
//            return y;
//        }
//  private:
//    double alpha, beta;
//};
//
//template< size_t n>
//struct dot_functor_T2d
//{
//    typedef thrust::tuple< double, int> Pair; 
//    dot_functor_T2d( double hx, double hy): hx(hx), hy(hy){}
//    __host__ __device__
//    double operator()( const double& x, const Pair& p) 
//    {
//        //generalized Multiplication
//        //laufender Index ist x-Index
//        return x*thrust::get<0>(p)*(2*get_j<n>( thrust::get<1>(p)) + 1)/hx
//                                  *(2*get_i<n>( thurst::get<1>(p)) + 1)/hy;
//    }
//    private:
//    double hx, hy;
//};
//
//
//template< size_t n>
//struct dsymv_functor_S2d
//{
//    typedef thrust::tuple< double, int> Pair;
//    dsymv_functor_S2d( double alpha, double beta): alpha(alpha), beta(beta) {}
//    __host__ __device__
//        double operator()( const double x, const Pair& p)
//        {
//            return alpha*x/(2*(thrust::get<1>(p)%n)+1)
//                  + beta*thrust::get<0>(p);
//        }
//  private:
//    double alpha, beta;
//};
//
//template< size_t n>
//struct dot_functor_S2d
//{
//    typedef thrust::tuple< double, int> Pair; 
//    dot_functor_S2d( double hx, double hy): hx(hx), hy(hy){}
//    __host__ __device__
//    double operator()( const double& x, const Pair& p) 
//    {
//        //generalized Multiplication
//        return x*thrust::get<0>(p)*hx/(2*get_j<n>(thrust::get<1>(p)) + 1)
//                                  *hy/(2*get_i<n>(thrust::get<1>(p)) + 1);
//    }
//    private:
//    double hx, hy;
//};
//}//namespace detail
//
//template< size_t n, class ThrustVector>
//struct BLAS2<T2D<n>, ThrustVector>
//{
//    typedef T2D<n> Matrix;
//    typedef ThrustVector Vector;
//    static void dsymv( double alpha, const Matrix& t, const ThrustVector& x, double beta, ThrustVector& y)
//    {
//        // x and y might be the same
//        if( alpha == 0)
//        {
//            if( beta == 1) 
//                return;
//            thrust::transform( y.begin(), y.end(), y.begin(), detail::daxpby_functor( 0, beta));
//            return;
//        }
//        thrust::transform( x.begin(), x.end(), 
//                          thrust::make_zip_iterator( 
//                                thrust::make_tuple( y.begin(), thrust::make_counting_iterator<int>(0)) ), 
//                          y.begin(),
//                          detail::dsymv_functor_T2d<n>( alpha/t.h(), beta)
//                          );
//    }
//    static void dsymv( const Matrix& t, const Vector& x, Vector& y)
//    {
//        dsymv( 1., t, x, 0., y);
//    }
//    static double ddot( const Vector& x, const Matrix& t, const Vector& y)
//    {
//        return thrust::inner_product(  x.begin(), x.end(), 
//                                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), thrust::make_counting_iterator(0)) ), 
//                                0.0,
//                                thrust::plus<double>(),
//                                detail::dot_functor_T2d<n>(t.h())
//                                );
//
//    }
//    static double ddot( const Matrix& t, const Vector& x) 
//    {
//        return ddot( x,t,x);
//    }
//
//};
//
//
//template< size_t n, class ThrustVector>
//struct BLAS2<S2D<n>, ThrustVector >
//{
//    typedef ThrustVector Vector;
//    typedef S2D<n> Matrix;
//    static void dsymv( double alpha, const Matrix& s, const Vector& x, double beta, Vector& y)
//    {
//        if( alpha == 0)
//        {
//            if( beta == 1) 
//                return;
//            thrust::transform( y.begin(), y.end(), y.begin(), detail::daxpby_functor( 0, beta));
//            return;
//        }
//        thrust::transform( x.begin(), x.end(), 
//                          thrust::make_zip_iterator( 
//                                thrust::make_tuple( y.begin(), thrust::make_counting_iterator(0)) ), 
//                          y.begin(),
//                          detail::dsymv_functor_S2d<n>( alpha*s.h(), beta)
//                          );
//    }
//    static void dsymv( const Matrix& s, const Vector& x, Vector& y)
//    {
//        dsymv( 1., s, x, 0., y);
//    }
//
//    static double ddot( const Vector& x, const Matrix& s, const Vector& y)
//    {
//        return thrust::inner_product(  x.begin(), x.end(), 
//                                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), thrust::make_counting_iterator(0)) ), 
//                                0.0,
//                                thrust::plus<double>(),
//                                detail::dot_functor_S2d<n>(s.h())
//                                );
//    }
//    static double ddot( const Matrix& s, const Vector& x)
//    {
//        return ddot( x, s, x);
//    }
//}; 
//} //nameapce dg
#endif //_DG_BLAS_PRECONDITIONER2D_
