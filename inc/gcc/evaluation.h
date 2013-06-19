#ifndef _DG_EVALUATION_
#define _DG_EVALUATION_

#include <iostream>
#include <vector>
#include <array>

#include "dlt.h"
#include "blas.h"

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

template< class Function, size_t n>
std::vector<std::array<double,n>> expand( Function& f, double a, double b, unsigned num_int)
{
    auto v = evaluate<Function,n> ( f, a, b, num_int);
    //multiply elements by forward
    double temp[n];
    for( unsigned k=0; k<num_int; k++)
    {
        for(unsigned i=0; i<n; i++)
        {
            temp[i] = 0;
            for( unsigned j=0; j<n; j++)
                temp[i] += dg::DLT<n>::forward[i][j]*v[k][j];
        }
        for( unsigned j=0; j<n; j++)
            v[k][j] = temp[j];
    }
    return v;
}

template< size_t n>
std::vector< double> evaluate_jump( const std::vector< std::array<double, n>>& v)
{
    //compute the interior jumps of a DG approximation
    unsigned N = v.size();
    std::vector<double> jump(N-1, 0.);
    for( unsigned i=0; i<N-1; i++)
        for( unsigned j=0; j<n; j++)
            jump[i] += v[i][j] - v[i+1][j]*( (j%2==0)?(1):(-1));
    return jump;
}


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


} //namespace dg

template <size_t n>
std::ostream& operator<<( std::ostream& os, const std::vector<std::array<double, n>>& v)
{
    unsigned N = v.size();
    for( unsigned i=0; i<N; i++)
    {
        for( unsigned j=0; j<n; j++)
            os << v[i][j] << " ";
        os << "\n";
    }
    return os;
}


#endif // _DG_EVALUATION_
