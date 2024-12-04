#include <iostream>
#include <iomanip>
#include "interpolation.h"
#include "grid.h"

int main()
{
    dg::Grid1d g1d( 1., 1.+2.*M_PI, 3, 10, dg::PER);
    dg::bc bcx[] = {dg::NEU, dg::DIR, dg::DIR_NEU, dg::NEU_DIR};
    std::cout << "Test the shift function. The numbers should be alternatively\
 2 and 2*M_PI read from up to down.\n";
    std::cout  << "PER NEU  DIR  DIR_NEU NEU_DIR\n";
    std::cout << std::boolalpha;
    double x = -10.*M_PI;
    for( int i =0; i<11; i++)
    {
        double x0 = x + i*2*M_PI;
        std::cout << std::setw( 8)<< x0 << " ";
        bool mirrored = false;
        dg::create::detail::shift( mirrored, x0, dg::PER, g1d.x0(), g1d.x1());
        if( false == mirrored && ( x0-2.*M_PI)<1e-15 ) std::cout << "PASSED ";
        else std::cout << "FAILED "<< mirrored<<" "<<x0<<" ";

        for( int j=0; j<4; j++)
        {
            x0 = x + i*2*M_PI;
            mirrored = false;
            dg::create::detail::shift( mirrored, x0, bcx[j], g1d.x0(), g1d.x1());
            std::cout << std::setw( 6)<<mirrored<<std::setw(8)<<x0 << " ";
        }
        std::cout <<std::endl;
    }
    std::cout << "Test 1d contains function\n";
    double x_in = M_PI, x_out = 3.*M_PI;
    assert( g1d.contains( x_in));
    assert( !g1d.contains( x_out));
    x_out = 0.;
    assert( !g1d.contains( x_out));
    std::cout << "PASSED!\n";
    std::cout << "Test 2d and 3d shift and contains functions\n";
    dg::Grid2d g2d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 3, 10, 10, dg::DIR, dg::DIR);
    dg::Grid2d g2d_test{ g2d.gx(),g2d.gy()};
    dg::Grid3d g3d( 1., 1.+2.*M_PI, 1., 1.+2.*M_PI, 0., 2.*M_PI, 3, 10, 10, 10,
        dg::DIR, dg::DIR, dg::PER);
    double y=0;
    for( int i=0; i<2; i++)
    {
        std::array<double,2> p = {0,y+i*2.*M_PI};
        std::cout << std::setw( 8)<< p[1] << " ";
        bool mirrored = false;
        dg::create::detail::shift( mirrored, p[0], g2d.bcx(), g2d.x0(), g2d.x1());
        dg::create::detail::shift( mirrored, p[1], g2d.bcy(), g2d.y0(), g2d.y1());
        std::cout << std::setw( 6)<<mirrored<<std::setw(8)<<p[0] <<std::setw(8)<<p[1] << "\n";
        std::array<double,3> q = {0,y+i*2.*M_PI, 2.*M_PI};
        mirrored = false;
        dg::create::detail::shift( mirrored, q[0], g3d.bcx(), g3d.x0(), g3d.x1());
        dg::create::detail::shift( mirrored, q[1], g3d.bcy(), g3d.y0(), g3d.y1());
        dg::create::detail::shift( mirrored, q[2], g3d.bcz(), g3d.z0(), g3d.z1());
        std::cout << std::setw( 15)<<mirrored<<std::setw(8)<<q[0]
                  << std::setw(  8)<<q[1] <<std::setw(8)<<q[2] << "\n";
    }
    std::cout << "Test 2d contains function\n";
    assert( g2d.contains( std::array{x_in, x_in}));
    std::array<double,2> p_out[] = {{0, M_PI}, {0,0}, {3*M_PI,3*M_PI},{M_PI,0}, {0,3*M_PI}};
    for( int i=0; i<4; i++)
        assert( !g2d.contains( p_out[i]));
    std::cout << "PASSED!\n";

    return 0;
}
