#ifndef _DG_DX_
#define _DG_DX_

#include <assert.h>

#include "blas.h"
#include "operators.h"
#include "projection_functions.h"
namespace dg
{

template<size_t n>
struct DX
{
    DX( double h = 2.){
        Operator<double, n> r( rirj);
        Operator<double, n> l( lilj);
        Operator<double, n> rl( rilj);
        Operator<double, n> d( pidxpj); 
        Operator<double, n> t( pipj_inv);
        //std::cout << h <<std::endl;
        //std::cout << " T is \n" <<t <<std::endl;
        t*=2./h;
        //std::cout << " T is \n" <<t <<std::endl;
        a = t*(d-r);
        b = t*rl;
    }
    const Operator<double,n>& get_a() const {return a;}
    const Operator<double,n>& get_b() const {return b;}
    private:
    Operator<double,n > a,b;
};
template <size_t n>
struct BLAS2< DX<n>, std::vector<std::array<double,n>>>
{
    typedef DX<n> Matrix;
    typedef std::vector<std::array<double,n>> Vector;
    static void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
    { //what is if x and y are the same??
        assert( &x != &y); 
        /*
            y[k] = alpha*(  A*x[k] + B*x[k+1]) + beta*y[k];
            y[N] = alpha*(  A*x[N] + B*x[0]  ) + beta*y[N];
        */
        const dg::Operator<double, n> & a = m.get_a();
        const dg::Operator<double, n> & b = m.get_b();
        const unsigned N = x.size();

        for( unsigned k=0; k<N-1; k++)
            for( unsigned i=0; i<n; i++)
            {
                y[k][i] = beta*y[k][i];
                for( unsigned j=0; j<n; j++)
                    y[k][i] += alpha*(a(i,j)*x[k][j] + b(i,j)*x[k+1][j]);
            }
        for( unsigned i=0; i<n; i++)
        {
            y[N-1][i] = beta*y[N-1][i];
            for( unsigned j=0; j<n; j++)
                y[N-1][i] += alpha*(a(i,j)*x[N-1][j] + b(i,j)*x[0][j]);
        }
    }
};
} //namespace dg

#endif //_DG_DX_
