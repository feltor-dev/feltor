#include <iostream>
#include <iomanip>

#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/evaluation.cuh"
#include "dg/rk.cuh"
#include "dg/xspacelib.cuh"
#include "dg/typedefs.cuh"

#include "shu.cuh"
#include "parameters.h"



using namespace std;
using namespace dg;
//const unsigned k=4;
const double Tmax = 0.01;
const double eps = 1e-14;
const unsigned n=1; //make error in space small
unsigned Nx = 200, Ny = Nx;

int main( int argc, char * argv[])
{

    std::cout << "k n dt Nx eps vort enstr energ\n";
    for(unsigned i=0; i<4;i++)
    {
        double dt = 1e-3/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        Grid<double> grid( 0, 1, 0, 1, n, Nx, Ny, dg::PER, dg::PER);
        DVec w2d( create::w2d(grid));
        dg::Lamb lamb( 0.5, 0.8, 0.1, 1.);
        HVec omega = evaluate ( lamb, grid);
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        Shu<DVec> shu( grid, 0, eps);
        AB< 1, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y0, y1, dt);
        y0.swap( y1); //y1 now contains value at zero time

        DVec stencil = evaluate( one, grid);
        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        double time = 0;
        unsigned step = 0;
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu, y0, y1, dt);
            y0.swap( y1); 
        }
        }
        catch( dg::Fail& fail) { 
            std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
            std::cerr << "Does Simulation respect CFL condition?\n";
        }
        std::cout << 1 <<" "<<n<<" "<<dt<<" "<<Nx<<" "<<eps<<" ";
        std::cout << fabs(blas2::dot( stencil , w2d, y1));
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, y1)-enstrophy)/enstrophy;
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, shu.potential())-energy)/energy <<"\n";
    }
    std::cout << std::endl;
    for(unsigned i=0; i<4;i++)
    {
        double dt = 1e-3/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        Grid<double> grid( 0, 1, 0, 1, n, Nx, Ny, dg::PER, dg::PER);
        DVec w2d( create::w2d(grid));
        dg::Lamb lamb( 0.5, 0.8, 0.1, 1.);
        HVec omega = evaluate ( lamb, grid);
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        Shu<DVec> shu( grid, 0, eps);
        AB< 2, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y0, y1, dt);
        y0.swap( y1); //y1 now contains value at zero time

        DVec stencil = evaluate( one, grid);
        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        double time = 0;
        unsigned step = 0;
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu, y0, y1, dt);
            y0.swap( y1); 
        }
        }
        catch( dg::Fail& fail) { 
            std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
            std::cerr << "Does Simulation respect CFL condition?\n";
        }
        std::cout << 2 <<" "<<n<<" "<<dt<<" "<<Nx<<" "<<eps<<" ";
        std::cout << fabs(blas2::dot( stencil , w2d, y1));
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, y1)-enstrophy)/enstrophy;
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, shu.potential())-energy)/energy <<"\n";
    }
    std::cout << std::endl;
    for(unsigned i=0; i<4;i++)
    {
        double dt = 1e-3/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        Grid<double> grid( 0, 1, 0, 1, n, Nx, Ny, dg::PER, dg::PER);
        DVec w2d( create::w2d(grid));
        dg::Lamb lamb( 0.5, 0.8, 0.1, 1.);
        HVec omega = evaluate ( lamb, grid);
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        Shu<DVec> shu( grid, 0, eps);
        AB< 3, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y0, y1, dt);
        y0.swap( y1); //y1 now contains value at zero time

        DVec stencil = evaluate( one, grid);
        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        double time = 0;
        unsigned step = 0;
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu, y0, y1, dt);
            y0.swap( y1); 
        }
        }
        catch( dg::Fail& fail) { 
            std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
            std::cerr << "Does Simulation respect CFL condition?\n";
        }
        std::cout << 3 <<" "<<n<<" "<<dt<<" "<<Nx<<" "<<eps<<" ";
        std::cout << fabs(blas2::dot( stencil , w2d, y1));
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, y1)-enstrophy)/enstrophy;
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, shu.potential())-energy)/energy <<"\n";
    }
    std::cout << std::endl;
    for(unsigned i=0; i<4;i++)
    {
        double dt = 1e-3/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        Grid<double> grid( 0, 1, 0, 1, n, Nx, Ny, dg::PER, dg::PER);
        DVec w2d( create::w2d(grid));
        dg::Lamb lamb( 0.5, 0.8, 0.1, 1.);
        HVec omega = evaluate ( lamb, grid);
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        Shu<DVec> shu( grid, 0, eps);
        AB< 4, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y0, y1, dt);
        y0.swap( y1); //y1 now contains value at zero time

        DVec stencil = evaluate( one, grid);
        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        double time = 0;
        unsigned step = 0;
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu, y0, y1, dt);
            y0.swap( y1); 
        }
        }
        catch( dg::Fail& fail) { 
            std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
            std::cerr << "Does Simulation respect CFL condition?\n";
        }
        std::cout << 4 <<" "<<n<<" "<<dt<<" "<<Nx<<" "<<eps<<" ";
        std::cout << fabs(blas2::dot( stencil , w2d, y1));
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, y1)-enstrophy)/enstrophy;
        std::cout << " "<<fabs(0.5*blas2::dot( y1, w2d, shu.potential())-energy)/energy <<"\n";
    }
    std::cout << std::endl;
    return 0;

}
