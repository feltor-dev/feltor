#include <iostream>
#include <complex>
#include "fma.h"

int main()
{
    double t = 3, f = 4, ff = 5;
    assert( DG_FMA( t,f,ff) == 17);
    {
    std::complex<double> z0( 1,2), z1( 3,4);
    assert( DG_FMA( t, z0, z1) == std::complex<double>( 6, 10));
    std::complex<double> z2( 5,6), z3;
    z3 = z0*z1+z2;
    assert( DG_FMA( z0, z1, z2) == z3);
    }
    {
    thrust::complex<double> z0( 1,2), z1( 3,4);
    assert( DG_FMA( t, z0, z1) == std::complex<double>( 6, 10));
    thrust::complex<double> z2( 5,6), z3;
    z3 = z0*z1+z2;
    assert( DG_FMA( z0, z1, z2) == z3);
    }
    std::cout << "ALL PASSED\n";

    return 0;
}
