#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#include "../preconditioner.h"

namespace dg{

template< size_t n>
struct daxpby_functor_T
{
    typedef thrust::pair< double, int> Pair;
    dsymv_functor_T( double alpha, double beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        double operator()(const double& x,  const Pair& p)
        {
            return alpha*x*(2.*thrust::get<1>(p)%n+1.)
                  + beta*thrust::get<0>(p);
        }
  private:
    double alpha, beta;
};

template< size_t n>
struct dot_functor_T
{
    typedef thrust::pair< double, int> Pair; 
    dot_functor_T( double h): h(h){}
    __host__ __device__
    double operator()( const double& x, const Pair& p) 
    {
        //generalized Multiplication
        return x*thrust::get<0>(p)/h*(2.*thrust::get<1>(p)%n + 1.);
    }

    private:
    double h;
};



template< size_t n>
struct daxpby_functor_S
{
    typedef thrust::tuple< double, int, double> Tuple;
    dsymv_functor_S( double alpha, double beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        double operator()( const double x, const Pair& p)
        {
            return alpha*x/(2.*thrust::get<1>(p)%n+1.)
                  + beta*thrust::get<0>(p);
        }
  private:
    double alpha, beta;
};
template< size_t n>
struct dot_functor_S
{
    typedef thrust::pair< double, int> Pair; 
    dot_functor_T( double h): h(h){}
    __host__ __device__
    double operator()( const double& x, const Pair& p) 
    {
        //generalized Multiplication
        return x*thrust::get<0>(p)*h/(2.*thrust::get<1>(p)%n + 1.);
    }

    private:
    double h;
};

template< size_t n>
struct BLAS2<T, thrust::device_vector<double> >
{
    typedef thrust::device_vector<double> Vector;
    typedef thrust::tuple< double, int, double> Tuple;
    static void dsymv( double alpha, const T& t, const Vector& x, double beta, Vector& y)
    {
        // x and y might be the same
        thrust::tranform( x.begin(), x.end(), 
                          thrust::make_zip_iterator( 
                                make_pair( y.begin(), thrust::make_counting_iterator(0)) ), 
                          y.begin(),
                          daxpy_functor_T<n>( alpha/t.h(), beta)
                          );
    }


    static void dsymv( const T& t, const Vector& x, Vector& y)
    {
        dsymv( 1., t, x, 0., y);
    }

    static double ddot( const Vector& x, const T& t, const Vector& y)
    {
        return thrust::inner_product(  x.begin(), x.end(), 
                                thrust::make_zip_iterator( make_pair( y.begin(), thrust::make_counting_iterator(0)) ), 
                                0.0,
                                thrust::plus<double>(),
                                dot_functor_T(t.h())
                                );

    }
    static double ddot( const T& t, const Vector& x) 
    {
        return ddot( x,t,x);
    }

};



template< size_t n>
struct BLAS2<S, thrust::device_vector<double> >
{
    typedef thrust::device_vector<double> Vector;
    static void dsymv( double alpha, const S& s, const Vector& x, double beta, Vector& y)
    {
        thrust::tranform( x.begin(), x.end(), 
                          thrust::make_zip_iterator( 
                                make_pair( y.begin(), thrust::make_counting_iterator(0)) ), 
                          y.begin(),
                          daxpy_functor_S<n>( alpha*s.h(), beta)
                          );
    }
    static void dsymv( const S& s, const Vector& x, Vector& y)
    {
        dsymv( 1., s, x, 0., y);
    }

    static double ddot( const Vector& x, const S& s, const Vector& y)
    {
        return thrust::inner_product(  x.begin(), x.end(), 
                                thrust::make_zip_iterator( make_pair( y.begin(), thrust::make_counting_iterator(0)) ), 
                                0.0,
                                thrust::plus<double>(),
                                dot_functor_S(s.h())
                                );
    }
    static double ddot( const S& s, const Vector& x)
    {
        return ddot( x, s, x);
    }
}; 
}
#endif //_DG_BLAS_PRECONDITIONER_
