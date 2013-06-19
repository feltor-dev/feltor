#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#include "../preconditioner.h"

namespace dg{

template< size_t n>
struct BLAS2<T, std::vector<std::array<double, n>>>
{
    typedef std::vector<std::array<double, n>> Vector;
    static void dsymv( double alpha, const T& t, const Vector& x, double beta, Vector& y)
    {
        // x and y might be the same
        unsigned N = x.size();
        for( unsigned i=0; i < N; i++)
            for( unsigned j=0; j<n; j++)
                y[i][j] = alpha*(2.*(double)j+1.)/t.h()*x[i][j] + beta*y[i][j];
    }
    static void dsymv( const T& t, const Vector& x, Vector& y)
    {
        dsymv( 1., t, x, 0., y);
    }

    static double ddot( const Vector& x, const T& t, const Vector& y)
    {
        double product = 0;
        unsigned N = x.size();
        for( unsigned i=0; i<N; i++)
            for( unsigned j=0; j<n; j++)
                product += (2.*(double)j+1.)/t.h()*x[i][j]*y[i][j];
        return product;
    }
    static double ddot( const T& t, const Vector& x) 
    {
        return ddot( x,t,x);
    }
};
template< size_t n>
struct BLAS2<S, std::vector<std::array<double, n>>>
{
    typedef std::vector<std::array<double, n>> Vector;
    static void dsymv( double alpha, const S& s, const Vector& x, double beta, Vector& y)
    {
        unsigned N = x.size();
        for( unsigned i=0; i < N; i++)
            for( unsigned j=0; j<n; j++)
                y[i][j] = alpha*s.h()/(2.*(double)j+1.)*x[i][j] + beta*y[i][j];
    }
    static void dsymv( const S& s, const Vector& x, Vector& y)
    {
        dsymv( 1., s, x, 0., y);
    }

    static double ddot( const Vector& x, const S& s, const Vector& y)
    {
        double product = 0;
        unsigned N = x.size();
        for( unsigned i=0; i<N; i++)
            for( unsigned j=0; j<n; j++)
                product += s.h()/(2.*(double)j+1.)*x[i][j]*y[i][j];
        return product;
    }
    static double ddot( const S& s, const Vector& x)
    {
        return ddot( x, s, x);
    }
}; 
}
#endif //_DG_BLAS_PRECONDITIONER_
