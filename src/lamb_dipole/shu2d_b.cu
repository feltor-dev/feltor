#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/algorithm.h"

#include "shu.cuh"

const double lx = 2.*M_PI;//*50.;
const double ly = 2.*M_PI;//*50.;

//const double U = 1.; //the dipole doesn't move with this velocity because box is not infinite
//const double R = 0.2*lx;
const double T = 2.;
////const double eps = 1e-7; //CG method

double D = 0.01;

//const double kx = 2.*M_PI* (double)m/lx;
//const double ky = 2.*M_PI* (double)m/ly;
//const double ksqr = (kx*kx+ky*ky) ;//4.*M_PI*M_PI*(1./lx/lx + 1./ly/ly);
const double kx = 1., ky = kx, ksqr = 2.;


double initial( double x, double y){ return 2.*sin(kx*x)*sin(ky*y);}
double solution( double x, double y){ return 2.*sin(kx*x)*sin(ky*y)*exp(-ksqr*D*T);}
double solution_phi( double x, double y){ return sin(kx*x)*sin(ky*y)*exp(-ksqr*D*T);}

//code for either lamb dipole or analytic sine function without graphics
int main()
{
    dg::Timer t;
    ////////////////////////////////////////////////////////////
    //std::cout << "Solve 2D incompressible NavierStokes with sin(x)sin(y) or Lamb dipole initial condition\n";
    //std::cout << "Type n, N and eps\n";
    //std::cin >> n >> Nx >>eps;
    //Ny = Nx;
    //std::cout << "Type diffusion constant!\n";
    //std::cin >> D;
    //std::cout << "# of Legendre coefficients: " << n<<std::endl;
    //std::cout << "# of grid cells:            " << Nx*Ny<<std::endl;
    std::cout << "# grid NT dt eps eps_V eps_omega eps_E eps\n";
    std::cout << "Diffusion " << D <<std::endl;

    ////////////////////////////////////////////////////////////
    for( unsigned n=2; n<3; n++)
    {
        std::cout << "P="<<n<<"\n";
        for(unsigned i=1; i<5; i++)
        {
            unsigned Nx = 8*pow(2,i), Ny = Nx;
            dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
            dg::DVec w2d( dg::create::weights(grid));

            double dx = lx/(double)Nx;
            double eps = 1e-1/pow(10, n)*pow(dx,n);
            unsigned NT = 4*(unsigned)(T*pow(2,n)/dx);
            //if( D!= 0)
                //NT = std::max((unsigned)(0.06*T*pow(4,n)/dx/dx), NT);
            const double dt = T/(double)NT;
            //cout << "Runge Kutta stages          " << k <<std::endl;
            //cout << "Timestep                    " << dt << std::endl;
            //cout << "# of steps                  " << NT <<std::endl;
            ////////////////////////////////////////////////////////////

            dg::DVec stencil = dg::evaluate( dg::one, grid);
            dg::DVec omega = dg::evaluate( initial, grid );
            const dg::DVec sol = dg::evaluate( solution, grid );
            const dg::DVec sol_phi = dg::evaluate( solution_phi, grid );

            dg::DVec y0( omega), y1( y0);
            //make solver and stepper
            shu::Shu<dg::DMatrix, dg::DVec> shu( grid, eps);
            shu::Diffusion<dg::DMatrix, dg::DVec> diffusion( grid, D);
            dg::Karniadakis< dg::DVec > karniadakis( y0, y0.size(), 1e-8);

            shu(0., y0, y1);
            double vorticity = dg::blas2::dot( stencil, w2d, sol);
            double enstrophy = 0.5*dg::blas2::dot( sol, w2d, sol);
            double energy =    0.5*dg::blas2::dot( sol, w2d, sol_phi) ;

            double time = 0;
            karniadakis.init( shu,diffusion, time, y0, dt);
            while( time < T)
            {
                //step

                t.tic();
                karniadakis.step( shu, diffusion, time, y0);
                t.toc();
                time += dt;
                //std::cout << "Time "<<time<< " "<<t.diff()<<"\n";
                if( fabs(dg::blas2::dot( w2d, y0)) > 1e16)
                {
                    std::cerr << "Sim unstable at time "<<time<<"!\n\n\n";
                    break;
                }
            }
            //std::cout << "Total simulation time:     "<<t.diff()<<"s\n";
            //std::cout << "Average Time for one step: "<<t.diff()/(double)NT<<"s\n";
            ////////////////////////////////////////////////////////////////////
            std::cout << Nx;
            std::cout << " "<<NT;
            std::cout << " "<<dt;
            std::cout << " "<<eps;
            std::cout << " "<<fabs(dg::blas2::dot( stencil, w2d, y0));
            std::cout << " "<<fabs(0.5*dg::blas2::dot( w2d, y0) - enstrophy)/enstrophy;
            std::cout << " "<<fabs(0.5*dg::blas2::dot( shu.potential(), w2d, y0) - energy)/energy<<" ";

            dg::blas1::axpby( 1., sol, -1., y0);
            std::cout << " "<<sqrt( dg::blas2::dot( w2d, y0))<< std::endl;
            //std::cout << "Relative distance to solution "<<sqrt( dg::blas2::dot( w2d, y0))/sqrt( dg::blas2::dot( w2d, sol)) << std::endl;

        }
        std::cout << std::endl;
    }
    //energy and enstrophy errrors are due to timestep only ( vorticity is exactly conserved)
    // k = 2 | p = 3
    // k = 3 | p = 4
    // k = 4 | p = 5

    //solution to sin(x)sin(y) 
    // n = 1 
    // n = 2 | p = 2
    // n = 3 | p = 2.6
    // n = 4 | p = 4

    return 0;

}
