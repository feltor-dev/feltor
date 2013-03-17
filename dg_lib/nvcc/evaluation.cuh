#ifndef _DG_EVALUATION_
#define _DG_EVALUATION_

#include "dgvec.cuh"
#include "dlt.h"

namespace dg
{

//some utility functions on the host
template< class Function, size_t n>
ArrVec1d< double, n> evaluate( Function& f, double a, double b, unsigned num_int)
{
    ArrVec1d< double, n> v(num_int);
    const double h = (b-a)/2./(double)num_int;
    double xp=1.;
    /* x = (b-a)/2N x' +a  maps the function to [0;2N]
      then x' goes through 1,3,5,...,2N-1
     */
    for( unsigned i=0; i<num_int; i++)
    {
        for( unsigned j=0; j<n; j++)
            v(i,j) = f( h*(xp + DLT<n>::abscissa[j])+a);
        xp+=2.;
    }
    return v;
}

template< class Function, size_t n>
ArrVec1d<double, n> expand( Function& f, double a, double b, unsigned num_int)
{
    ArrVec1d<double, n> v = evaluate<Function,n> ( f, a, b, num_int);
    //multiply elements by forward
    double temp[n];
    for( unsigned k=0; k<num_int; k++)
    {
        for(unsigned i=0; i<n; i++)
        {
            temp[i] = 0;
            for( unsigned j=0; j<n; j++)
                temp[i] += dg::DLT<n>::forward[i][j]*v(k,j);
        }
        for( unsigned j=0; j<n; j++)
            v(k,j) = temp[j];
    }
    return v;
}

template< size_t n>
thrust::host_vector< double> evaluate_jump( const ArrVec1d<double, n>& v)
{
    //compute the interior jumps of a DG approximation
    unsigned N = v.size();
    thrust::host_vector<double> jump(N-1, 0.);
    for( unsigned i=0; i<N-1; i++)
        for( unsigned j=0; j<n; j++)
            jump[i] += v(i,j) - v(i+1,j)*( (j%2==0)?(1):(-1));
    return jump;
}


}//namespace dg

#endif //_DG_EVALUATION
