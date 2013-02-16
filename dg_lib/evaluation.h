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



}


#endif // _DG_EVALUATION_
