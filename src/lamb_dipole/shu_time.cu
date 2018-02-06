#include <iostream>
#include <iomanip>

#include "dg/backend/timer.cuh"
#include "dg/algorithm.h"
#include "dg/functors.h"
#include "dg/backend/evaluation.cuh"
#include "dg/runge_kutta.h"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/typedefs.cuh"

#include "shu.cuh"
#include "parameters.h"



using namespace std;
using namespace dg;
//const unsigned k=4;
const double Tmax = 0.01;
const double eps = 1e-14;
const unsigned n=1; //make error in space small
unsigned Nx = 100, Ny = Nx;

int main( int argc, char * argv[])
{


    double dt0; 
    std::cout << "type dt0 (1e-3)!\n";
    std::cin >> dt0;
    std::cout << "k n dt Nx eps vort enstr energy\n";
    Grid2d grid( 0, 1, 0, 1, n, Nx, Ny, dg::PER, dg::PER);
    DVec w2d( create::weights(grid));
    dg::Lamb lamb( 0.5, 0.8, 0.1, 1.);
    const HVec omega = evaluate ( lamb, grid);
    Shu<dg::DMatrix, dg::DVec> shu( grid, eps);
    const DVec stencil = evaluate( one, grid);
    for(unsigned i=0; i<6;i++)
    {
        double dt = dt0/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        AB< 1, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y1);

        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu, y1);
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
    for(unsigned i=0; i<6;i++)
    {
        double dt = dt0/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        AB< 2, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y1);

        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu, y1);
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
    for(unsigned i=0; i<6;i++)
    {
        double dt = dt0/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        AB< 3, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y1);

        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu,y1);
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
    for(unsigned i=0; i<6;i++)
    {
        double dt = dt0/pow(2,i);
        unsigned NT = (unsigned)(Tmax/dt);
        //initiate solver 
        DVec y0( omega ), y1( y0);
        //make solver and stepper
        AB< 4, DVec > ab( y0);
        ab.init( shu, y0, dt);
        ab( shu, y1);

        double vorticity = blas2::dot( stencil, w2d, y1);
        double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
        double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
        /////////////////////////////////////////////////////////////////
        try{
        for( unsigned i=0; i<NT; i++)
        {
            ab( shu, y1);
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
