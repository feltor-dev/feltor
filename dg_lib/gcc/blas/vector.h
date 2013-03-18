#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#include <vector>
#include <array>

#include "../blas.h"

namespace dg{

template <size_t n>
struct BLAS1<std::vector<std::array<double,n>>>
{
    typedef std::vector<std::array<double,n>> Vector;
    static double ddot( const Vector& x, const Vector& y)
    {
        assert( x.size() == y.size());
        double sum = 0;
        double s = 0;
        for( unsigned i=0; i<x.size(); i++)
        {
            s=0;
            for( unsigned j=0; j<n; j++)
                s+= x[i][j]*y[i][j];
            sum +=s;
        }
        return sum;
    }
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        assert( x.size() == y.size());
        if( alpha == 0.)
        {
            if( beta == 1.) return;
            for( unsigned i=0; i<x.size(); i++)
                for( unsigned j=0; j<n; j++)
                    y[i][j] = beta*y[i][j];
            return; 
        }
        for( unsigned i=0; i<x.size(); i++)
            for( unsigned j=0; j<n; j++)
                y[i][j] = alpha*x[i][j]+beta*y[i][j];
    }
};

} //namespace dg
#endif // _DG_BLAS_VECTOR
