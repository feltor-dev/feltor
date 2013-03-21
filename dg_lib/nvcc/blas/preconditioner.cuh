#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "../preconditioner.cuh"
#include "thrust_vector.cuh" //load thrust_vector BLAS1 routines
#include "../blas.h"

namespace dg{

namespace detail{

/*
template< size_t n>
struct dsymv_functor_T
{
    typedef thrust::tuple< double, int> Pair;
    dsymv_functor_T( double alpha, double beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        double operator()(const double& x,  const Pair& p)
        {
            double y = alpha*x *(2.* (thrust::get<1>(p)%n)+1.)
                        + beta*thrust::get<0>(p);
            return y;
        }
  private:
    double alpha, beta;
};

template< size_t n>
struct dot_functor_T
{
    typedef thrust::tuple< double, int> Pair; 
    dot_functor_T( double h): h(h){}
    __host__ __device__
    double operator()( const double& x, const Pair& p) 
    {
        //generalized Multiplication
        return x*thrust::get<0>(p)/h*(2.*(thrust::get<1>(p)%n) + 1.);
    }

    private:
    double h;
};
*/
template< class Derived> 
struct dsymv_functor
{
    typedef thrust::tuple< double, int> Pair;
    dsymv_functor_T( double alpha, double beta, const Derived& p ): p_(p), alpha(alpha), beta(beta) {}
    __host__ __device__
        double operator()(const double& x,  const Pair& p)
        {
            double y = alpha*x *p_(thrust::get<1>(p))
                        + beta*thrust::get<0>(p);
            return y;
        }
  private:
    Derived p_;
    double alpha, beta;
};

template< class Derived> 
struct dot_functor
{
    typedef thrust::tuple< double, int> Pair; 
    dot_functor_T( const Derived& p): p_(p){}
    __host__ __device__
    double operator()( const double& x, const Pair& p) 
    {
        //generalized Multiplication
        return x*thrust::get<0>(p)*p_(thrust::get<1>(p));
    }

  private:
    const Derived p_;
};

/*

template< size_t n>
struct dsymv_functor_S
{
    typedef thrust::tuple< double, int> Pair;
    dsymv_functor_S( double alpha, double beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        double operator()( const double x, const Pair& p)
        {
            return alpha*x/(2.*(thrust::get<1>(p)%n)+1.)
                  + beta*thrust::get<0>(p);
        }
  private:
    double alpha, beta;
};

template< size_t n>
struct dot_functor_S
{
    typedef thrust::tuple< double, int> Pair; 
    dot_functor_S( double h): h(h){}
    __host__ __device__
    double operator()( const double& x, const Pair& p) 
    {
        //generalized Multiplication
        return x*thrust::get<0>(p)*h/(2.*(thrust::get<1>(p)%n) + 1.);
    }

    private:
    double h;
};
*/
}//namespace detail

template< class Derived, class ThrustVector>
struct BLAS2<DiagonalPreconditioner<Derived>, ThrustVector>
{
    typedef Derived Matrix;
    typedef ThrustVector Vector;
    static void dsymv( double alpha, const Matrix& t, const ThrustVector& x, double beta, ThrustVector& y)
    {
        // x and y might be the same
        if( alpha == 0)
        {
            if( beta == 1) 
                return;
            thrust::transform( y.begin(), y.end(), y.begin(), detail::daxpby_functor( 0, beta));
            return;
        }
        thrust::transform( x.begin(), x.end(), 
                          thrust::make_zip_iterator( 
                                thrust::make_tuple( y.begin(), thrust::make_counting_iterator<int>(0)) ), 
                          y.begin(),
                          detail::dsymv_functor<Derived>( alpha, beta, t)
                          );
    }
    static void dsymv( const Matrix& t, const Vector& x, Vector& y)
    {
        dsymv( 1., t, x, 0., y);
    }
    static double ddot( const Vector& x, const Matrix& t, const Vector& y)
    {
        return thrust::inner_product(  x.begin(), x.end(), 
                                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), thrust::make_counting_iterator(0)) ), 
                                0.0,
                                thrust::plus<double>(),
                                detail::dot_functor<Derived>( t)
                                );

    }
    static double ddot( const Matrix& t, const Vector& x) 
    {
        return ddot( x,t,x);
    }

};


/*
template< size_t n, class ThrustVector>
struct BLAS2<S<n>, ThrustVector >
{
    typedef ThrustVector Vector;
    typedef S<n> Matrix;
    static void dsymv( double alpha, const Matrix& s, const Vector& x, double beta, Vector& y)
    {
        if( alpha == 0)
        {
            if( beta == 1) 
                return;
            thrust::transform( y.begin(), y.end(), y.begin(), detail::daxpby_functor( 0, beta));
            return;
        }
        thrust::transform( x.begin(), x.end(), 
                          thrust::make_zip_iterator( 
                                thrust::make_tuple( y.begin(), thrust::make_counting_iterator(0)) ), 
                          y.begin(),
                          detail::dsymv_functor_S<n>( alpha*s.h(), beta)
                          );
    }
    static void dsymv( const Matrix& s, const Vector& x, Vector& y)
    {
        dsymv( 1., s, x, 0., y);
    }

    static double ddot( const Vector& x, const Matrix& s, const Vector& y)
    {
        return thrust::inner_product(  x.begin(), x.end(), 
                                thrust::make_zip_iterator( thrust::make_tuple( y.begin(), thrust::make_counting_iterator(0)) ), 
                                0.0,
                                thrust::plus<double>(),
                                detail::dot_functor_S<n>(s.h())
                                );
    }
    static double ddot( const Matrix& s, const Vector& x)
    {
        return ddot( x, s, x);
    }
}; 
*/
} //nameapce dg
#endif //_DG_BLAS_PRECONDITIONER_
