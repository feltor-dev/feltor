#include <iostream>
#include <iomanip>
#include "grid.h"

int main()
{
    dg::Grid1d g1d( 1., 1.+2.*M_PI, 3, 10, dg::PER);
    double x = 0.-10.*M_PI;
    dg::bc bcx[] = {dg::NEU, dg::DIR, dg::DIR_NEU, dg::NEU_DIR};
    std::cout << "Test the shift function. The numbers should be alternatively 2 and 2*M_PI read from up to down.\n";
    std::cout  << "PER NEU  DIR  DIR_NEU NEU_DIR\n";
    std::cout << std::boolalpha;
    for( int i =0; i<11; i++)
    {
        std::array<double,1> x0 = {x + i*2*M_PI};
        std::cout << std::setw( 8)<< x0[0] << " ";
        bool mirrored = false;
        g1d.shift( mirrored, x0);
        if( false == mirrored && ( x0[0]-2.*M_PI)<1e-15 ) std::cout << "PASSED ";
        else std::cout << "FAILED "<< mirrored<<" "<<x0[0]<<" ";

        for( int j=0; j<4; j++)
        {
            x0 = {x + i*2*M_PI};
            mirrored = false;
            g1d.shift( mirrored, x0, {bcx[j]});
            std::cout << std::setw( 6)<<mirrored<<std::setw(8)<<x0[0] << " ";
        }
        std::cout <<std::endl;
    }
    std::cout << "Test 2d and 3d shift functions\n";
    dg::Grid2d g2d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10, dg::DIR, dg::DIR);
    dg::Grid2d g2d_test( {1., 1.+2.*M_PI,3,10,dg::DIR},{ 1., 1.+2.*M_PI, 10, dg::DIR});
    dg::Grid3d g3d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0., 2.*M_PI, 3, 10, 10, 10, dg::DIR, dg::DIR, dg::PER);
    double y=0;
    for( int i=0; i<2; i++)
    {
        std::array<double,2> p = {0,y+i*2.*M_PI};
        std::cout << std::setw( 8)<< p[1] << " ";
        bool mirrored = false;
        g2d.shift( mirrored, p);
        std::cout << std::setw( 6)<<mirrored<<std::setw(8)<<p[0] <<std::setw(8)<<p[1] << "\n";
        std::array<double,3> q = {0,y+i*2.*M_PI, 2.*M_PI};
        mirrored = false;
        g3d.shift( mirrored, q);
        std::cout << std::setw( 15)<<mirrored<<std::setw(8)<<q[0] <<std::setw(8)<<q[1] <<std::setw(8)<<q[2] << "\n";
    }

    return 0;
}
