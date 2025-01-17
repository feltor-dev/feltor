#include <iostream>

#include "dg/algorithm.h"
#include "outer.h"


int main()
{
    dg::Grid1d gx( 0,1 , 3, 8);
    dg::Grid1d gy( 0,2 , 4, 7);
    dg::Grid1d gz( 1,2 , 5, 7);
    dg::Grid2d g2d ( gx, gy);
    dg::Grid3d g3d ( gx, gy, gz);

    dg::HVec w1dX = dg::create::weights( gx);
    dg::HVec w1dY = dg::create::weights( gy);
    dg::HVec w1dZ = dg::create::weights( gz);
    dg::HVec w2d =  dg::create::weights( g2d);
    dg::HVec w3d =  dg::create::weights( g3d);
    dg::HVec w2d_test (w2d), w3d_test(w3d);
    dg::mat::outer_product( w1dX, w1dY, w2d_test);
    dg::mat::outer_product( w1dX, w1dY, w1dZ, w3d_test);
    for( unsigned i=0; i<w2d.size(); i++)
        if( fabs(w2d[i] - w2d_test[i]) > 1e-15)
            std::cout << "2d TEST FAILED "<<w2d[i]<<" "<<w2d_test[i]<<"\n";
    for( unsigned i=0; i<w3d.size(); i++)
        if( fabs(w3d[i] - w3d_test[i]) > 1e-15)
            std::cout << "3d TEST FAILED "<<w3d[i]<<" "<<w3d_test[i]<<"\n";
    std::cout << "TEST PASSED (if not stated otherwise)!\n";
    return 0;
}
