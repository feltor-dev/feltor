#ifndef _DG_BLAS_LAPLACE_
#define _DG_BLAS_LAPLACE_

#include <vector>
#include <array>

#include "../blas.h"
#include "../laplace.h"

namespace dg{

template <size_t n>
struct BLAS2< Laplace_Dir<n>, std::vector<std::array<double,n>>>
{
    typedef Laplace_Dir<n> Matrix;
    typedef std::vector<std::array<double,n>> Vector;
    static void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
    {
        assert( &x != &y); 
        /*
            y[0] = alpha*(              Ap*x[0]+ Bp*x[1] ) + beta*y[0];
            y[1] = alpha*(  Bp^T*x[0] + A*x[1]+ B*x[2]  ) + beta*y[1];
            y[k] = alpha*( B^T*x[k-1] + A*x[k] + B*x[k+1]) + beta*y[k];
          y[N-1] = alpha*( B^T*x[N-2] + A*x[N-1]         ) + beta*y[N-1];
        */
        const dg::Operator<double, n> & a = m.get_a();
        const dg::Operator<double, n> & b = m.get_b();
        const unsigned N = x.size();

        const dg::Operator<double, n> & ap = m.get_ap();
        const dg::Operator<double, n> & bp = m.get_bp();
        for( unsigned i=0; i<n; i++)
        {
            y[0][i] = beta*y[0][i];
            for( unsigned j=0; j<n; j++)
                y[0][i] += alpha*( ap(i,j)*x[0][j] + bp(i,j)*x[1][j]);
        }
        for( unsigned i=0; i<n; i++)
        {
            y[1][i] = beta*y[1][i];
            for( unsigned j=0; j<n; j++)
                y[1][i] += alpha*( bp(j,i)*x[0][j] + a(i,j)*x[1][j] + b(i,j)*x[2][j]);
        }
        for( unsigned k=2; k<N-1; k++)
            for( unsigned i=0; i<n; i++)
            {
                y[k][i] = beta*y[k][i];
                for( unsigned j=0; j<n; j++)
                    y[k][i] += alpha*(b(j,i)*x[k-1][j] + a(i,j)*x[k][j] + b(i,j)*x[k+1][j]);
            }
        for( unsigned i=0; i<n; i++)
        {
            y[N-1][i] = beta*y[N-1][i];
            for( unsigned j=0; j<n; j++)
                y[N-1][i] += alpha*(b(j,i)*x[N-2][j] + a(i,j)*x[N-1][j] );
        }
    }
    static void dsymv( const Matrix& m, const Vector& x, Vector& y)
    {
        dsymv( 1., m, x, 0., y);
    }
};

template <size_t n>
struct BLAS2< Laplace<n>, std::vector<std::array<double,n>>>
{
    typedef Laplace<n> Matrix;
    typedef std::vector<std::array<double,n>> Vector;
    static void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
    { //what is if x and y are the same??
        assert( &x != &y); 
        /*
            y[0] = alpha*( B^T*x[N-1] + A*x[0] + B*x[1]  ) + beta*y[0];
            y[k] = alpha*( B^T*x[k-1] + A*x[k] + B*x[k+1]) + beta*y[k];
            y[N] = alpha*( B^T*x[N-1] + A*x[N] + B*x[0]  ) + beta*y[N];
        */
        const dg::Operator<double, n> & a = m.get_a();
        const dg::Operator<double, n> & b = m.get_b();
        const unsigned N = x.size();

        for( unsigned i=0; i<n; i++)
        {
            y[0][i] = beta*y[0][i];
            for( unsigned j=0; j<n; j++)
                y[0][i] += alpha*(b(j,i)*x[N-1][j] + a(i,j)*x[0][j] + b(i,j)*x[1][j]);
        }
        for( unsigned k=1; k<N-1; k++)
            for( unsigned i=0; i<n; i++)
            {
                y[k][i] = beta*y[k][i];
                for( unsigned j=0; j<n; j++)
                    y[k][i] += alpha*(b(j,i)*x[k-1][j] + a(i,j)*x[k][j] + b(i,j)*x[k+1][j]);
            }
        for( unsigned i=0; i<n; i++)
        {
            y[N-1][i] = beta*y[N-1][i];
            for( unsigned j=0; j<n; j++)
                y[N-1][i] += alpha*(b(j,i)*x[N-2][j] + a(i,j)*x[N-1][j] + b(i,j)*x[0][j]);
        }
    }
    static void dsymv( const Matrix& m, const Vector& x, Vector& y)
    { //what is if x and y are the same??
        dsymv( 1., m, x, 0., y);
    }
};
} //namespace dg

#endif //_DG_BLAS_LAPLACE_
