#include <iostream>
#include <vector>
#include <array>

#include "operators.h"
#include "laplace.h"
#include "dlt.h"

#define P 4
typedef std::vector<std::array<double, P>> Vector;
typedef dg::Laplace_per<P> Matrix;

template < >
class CG_BLAS1<Vector>>
{
    double ddot( const Vector& x, const Vector& y)
    {
        double sum = 0;
        double s = 0;
        for( unsigned i=0; i<x.size(); i++)
        {
            s=0;
            for( unsigned j=0; j<P; j++)
                s+= x[i][j]*y[i][j];
            sum +=s;
        }
    }
    void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        for( unsigned i=0; i<x.size(); i++)
            for( unsigned j=0; j<P; j++)
                y[i][j]+= alpha*x[i][j]+beta*y[i][j];
    }

};



template <>
class CG_BLAS2< Matrix, Vector>
{
    void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
    {
        /*
            y[0] = alpha*( B^T*x[N]   + A*x[0] + B*x[1]  ) + beta*y[0];
            y[k] = alpha*( B^T*x[k-1] + A*x[k] + B*x[k+1]) + beta*y[k];
            y[N] = alpha*( B^T*x[N-1] + A*x[N] + B*x[0]  ) + beta*y[N];
        */
        const Operator<double, P> & a = m.get_a();
        const Operator<double, P> & b = m.get_b();
        const unsigned N = x.size();

        for( unsigned i=0; i<P; i++)
        {
            y[0][i] = beta*y[0][j];
            for( unsigned j=0; j<P; j++)
                y[0][i] += beta*(b(j,i)*x[N][j] + a(i,j)*x[0][j] + b(i,j)*x[1][j]);
        }
        for( unsigned k=1; k<N-1; k++)
            for( unsigned i=0; i<P; i++)
            {
                y[k][i] = beta*y[k][j];
                for( unsigned j=0; j<P; j++)
                    y[k][i] += beta*(b(j,i)*x[k-1][j] + a(i,j)*x[k][j] + b(i,j)*x[k+1][j]);
            }
        for( unsigned i=0; i<P; i++)
        {
            y[N][i] = beta*y[N][j];
            for( unsigned j=0; j<P; j++)
                y[N][i] += beta*(b(j,i)*x[N-1][j] + a(i,j)*x[N][j] + b(i,j)*x[0][j]);
        }
    }
};


using namespace std;
using namespace dg;
int main()
{
    Laplace_per<P> l;

    return 0;
}
