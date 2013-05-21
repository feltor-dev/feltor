#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include "grid.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"


namespace dg
{

typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> HVec;


template< class Function, size_t n>
HVec evaluate( Function& f, const Grid<double,n>& g)
{
    return (evaluate<Function, n>( f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};
template< size_t n>
HVec evaluate( double(f)(double, double), const Grid<double,n>& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return (evaluate<double(&)(double, double), n>( *f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};
}//namespace dg

#endif // _DG_XSPACELIB_CUH_
