#ifndef _DG_EVALUATION_
#define _DG_EVALUATION_

#include "dlt.h"
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

enum Sign{ XSPACE, LSPACE};

template< size_t n>
double square_norm( std::vector<std::array<double, n>> v, enum Sign s)
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
