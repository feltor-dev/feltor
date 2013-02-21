#ifndef _DG_LAPLACE_
#define _DG_LAPLACE_

#include "blas.h"
#include "projection_functions.h"
#include "operators.h"

namespace dg
{

template<size_t n>
class Laplace
{
  public:
    Laplace( double h = 2.);
    const Operator<double,n>& get_a() const {return a;}
    const Operator<double,n>& get_b() const {return b;}
  private:
    Operator<double, n> a,b;

};

template<size_t n>
Laplace<n>::Laplace( double h) 
{
    Operator<double, n> l( lilj);
    Operator<double, n> r( rirj);
    Operator<double, n> lr( lirj);
    Operator<double, n> rl( rilj);
    Operator<double, n> d( pidxpj);
    Operator<double, n> t( pipj_inv);
    t *= 2./h;

    std::cout << t <<std::endl;
    //std::cout << d << std::endl<< l<<std::endl;
    //std::cout << "(d+l)T(d+l)^T \n";
    //std::cout << (d+l)*t*(d+l).transpose()<<std::endl;
    //std::cout << lr*t*rl<<std::endl;
    a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    b = -((d+l)*t*rl+rl);
};

template <size_t n>
struct BLAS2< Laplace<n>, std::vector<std::array<double,n>>>
{
    typedef Laplace<n> Matrix;
    typedef std::vector<std::array<double,n>> Vector;
    static void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
    {
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
};

template<size_t n>
class Laplace_Dir
{
  public:
    Laplace_Dir( double h = 2.);
    const Operator<double,n>& get_a() const {return a;}
    const Operator<double,n>& get_b() const {return b;}
    const Operator<double,n>& get_ap() const {return ap;}
    const Operator<double,n>& get_bp() const {return bp;}
  private:
    Operator<double, n> a,b;
    Operator<double, n> ap,bp;

};

template<size_t n>
Laplace_Dir<n>::Laplace_Dir( double h) 
{
    Operator<double, n> l( lilj);
    Operator<double, n> r( rirj);
    Operator<double, n> lr( lirj);
    Operator<double, n> rl( rilj);
    Operator<double, n> d( pidxpj);
    Operator<double, n> s( pipj);
    Operator<double, n> t( pipj_inv);
    t *= 2./h;

    a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    b = -((d+l)*t*rl+rl);
    ap = d*t*d.transpose() + l + r;
    bp = -((d+l)*t*rl + rl);
};


template <size_t n>
struct BLAS2< Laplace_Dir<n>, std::vector<std::array<double,n>>>
{
    typedef Laplace<n> Matrix;
    typedef std::vector<std::array<double,n>> Vector;
    static void dsymv( double alpha, const Matrix& m, const Vector& x, double beta, Vector& y)
    {
        /*
            y[0] = alpha*(              Ap*x[0]+ Bp*x[1] ) + beta*y[0];
            y[1] = alpha*(  Bp^T*x[0] + Ap*x[1]+ B*x[2]  ) + beta*y[1];
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
};
} //namespace dg

#endif // _DG_LAPLACE_
