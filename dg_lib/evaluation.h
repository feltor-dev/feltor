#ifndef _DG_EVALUATION_
#define _DG_EVALUATION_

#include "dlt.h"
#include "blas.h"
#include <vector>
#include <array>
namespace dg
{

template< class Function, size_t n>
std::vector<std::array<double,n>> evaluate( Function& f, double a, double b, unsigned num_int)
{
    std::vector<std::array<double,n>> v(num_int);
    const double h = (b-a)/2./(double)num_int;
    double xp=1.;
    /* x = (b-a)/2N x' +a  maps the function to [0;2N]
      then x' goes through 1,3,5,...,2N-1
     */
    for( unsigned i=0; i<num_int; i++)
    {
        for( unsigned j=0; j<n; j++)
            v[i][j] = f( h*(xp + DLT<n>::abscissa[j])+a);
        xp+=2.;
    }
    return v;
}

struct T{
    T( double h = 2.):h_(h){}
    const double& h() const {return h_;}
  private:
    double h_;
};
struct S{
    S( double h = 2.):h_(h){}
    const double& h() const {return h_;}
  private:
    double h_;
};
template< size_t n>
struct BLAS2<T, std::vector<std::array<double, n>>>
{
    typedef std::vector<std::array<double, n>> Vector;
    static void dsymv( double alpha, const T& t, const Vector& x, double beta, Vector& y)
    {
        unsigned N = x.size();
        for( unsigned i=0; i < N; i++)
            for( unsigned j=0; j<n; j++)
                y[i][j] = alpha*(2.*(double)j+1.)/t.h()*x[i][j] + beta*y[i][j];
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

    static double ddot( const Vector& x, const S& s, const Vector& y)
    {
        double product = 0;
        unsigned N = x.size();
        for( unsigned i=0; i<N; i++)
            for( unsigned j=0; j<n; j++)
                product += s.h()/(2.*(double)j+1.)*x[i][j]*y[i][j];
        return product;
    }
}; //redundant code LSPACE does the same!
enum Space{ XSPACE, LSPACE};

template < size_t n >
struct BLAS2<Space, std::vector<std::array<double, n>>>
{
    typedef std::vector<std::array<double, n>> Vector;
    static void dsymv( double alpha, const Space& s, const Vector& x, double beta, Vector& y)
    {
        unsigned N = x.size();
        if( s == XSPACE)
        {
            for( unsigned i=0; i < N; i++)
                for( unsigned j=0; j<n; j++)
                    y[i][j] = alpha*DLT<n>::weight[j]*x[i][j] + beta*y[i][j];
        }
        else
        {
            for( unsigned i=0; i < N; i++)
                for( unsigned j=0; j<n; j++)
                    y[i][j] = alpha*2./(2.*(double)j+1.)*x[i][j] + beta*y[i][j];
        }
    }

    static double ddot( const Vector& x, const Space& s, const Vector& y)
    {
        double norm=0;
        unsigned N = x.size();
        if( s == XSPACE)
        {
            for( unsigned i=0; i<N; i++)
                for( unsigned j=0; j<n; j++)
                    norm += DLT<n>::weight[j]*x[i][j]*y[i][j];
        }
        else
        {
            for( unsigned i=0; i<N; i++)
                for( unsigned j=0; j<n; j++)
                    norm += 2./(2.*(double)j+1.)*x[i][j]*y[i][j];
        }
        return norm;
    }
};
//compute square norm on [0,2N]
//correct normalisation is (b-a)/2N
template< size_t n>
double square_norm( const std::vector<std::array<double, n>>& v, enum Space s)
{
    double norm=0;
    unsigned N = v.size();
    if( s == XSPACE)
    {
        for( unsigned i=0; i<N; i++)
            for( unsigned j=0; j<n; j++)
                norm += DLT<n>::weight[j]*v[i][j]*v[i][j];
    }
    else
    {
        for( unsigned i=0; i<N; i++)
            for( unsigned j=0; j<n; j++)
                norm += 2./(2.*(double)j+1.)*v[i][j]*v[i][j];
    }
    return norm;
}






}


#endif // _DG_EVALUATION_
