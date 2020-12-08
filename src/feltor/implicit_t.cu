#include <iostream>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/json_utilities.h"
#include "parameters.h"

#define DG_MANUFACTURED
#define FELTORPARALLEL 1
#define FELTORPERP 1
#include "manufactured.h"
#include "implicit.h"


int main( int argc, char* argv[])
{
    Json::Value js, gs;
    if( argc == 1)
        file::file2Json( "input.json", js, file::comments::are_forbidden);
    else if( argc == 2)
        file::file2Json( argv[1], js, file::comments::are_forbidden);
    else
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile]\n";
        return -1;
    }
    const feltor::Parameters p( js, file::error::is_throw);
    p.display( std::cout);
    const double R_0 = 10;
    const double I_0 = 20; //q factor at r=1 is I_0/R_0
    const double a  = 1; //small radius
    dg::CylindricalGrid3d grid( R_0-a, R_0+a, -a, a, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::DVec w3d = dg::create::volume( grid);

    //create RHS
    std::cout << "Initialize explicit" << std::endl;
    dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    std::cout << "Initialize implicit" << std::endl;
    feltor::Implicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec > im( grid, p, mag);
    feltor::FeltorSpecialSolver<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> solver( grid, p, mag);
    double alpha = -p.dt, time = 0.237; //Sin(3*Pi*t)
    feltor::manufactured::Snehat snehat{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1], alpha};
    feltor::manufactured::SNihat snihat{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1], alpha};
    feltor::manufactured::SWehat swehat{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1], alpha};
    feltor::manufactured::SWihat swihat{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1], alpha};
    feltor::manufactured::A aa{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ne ne{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ni ni{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::We we{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Wi wi{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ue ue{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ui ui{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    dg::DVec R = dg::pullback( dg::cooX3d, grid);
    dg::DVec Z = dg::pullback( dg::cooY3d, grid);
    dg::DVec P = dg::pullback( dg::cooZ3d, grid);
    dg::DVec zer = dg::evaluate( dg::zero, grid);
    std::array<dg::DVec,2> w{zer,zer}, sol_w{w};
    // ... init y to 0, rhs = rhs function and solutions to solutions
    std::array<std::array<dg::DVec,2>,2> y{w,w}, rhs( y), sol{y};
    dg::DVec apar{R}, sol_apar{apar};

    dg::blas1::evaluate( rhs[0][0], dg::equals(), snehat, R,Z,P,time);
    dg::blas1::evaluate( rhs[0][1], dg::equals(), snihat, R,Z,P,time);
    dg::blas1::plus( rhs[0], -1);
    dg::blas1::evaluate( rhs[1][0], dg::equals(), swehat, R,Z,P,time);
    dg::blas1::evaluate( rhs[1][1], dg::equals(), swihat, R,Z,P,time);
    dg::blas1::evaluate( apar, dg::equals(), aa, R,Z,P,time);

    solver.solve( alpha, im, time, y, rhs);

    dg::blas1::plus( y[0], +1); //we solve for n-1
    dg::blas1::evaluate( sol[0][0], dg::equals(), ne, R,Z,P,time);
    dg::blas1::evaluate( sol[0][1], dg::equals(), ni, R,Z,P,time);
    dg::blas1::evaluate( sol[1][0], dg::equals(), we, R,Z,P,time);
    dg::blas1::evaluate( sol[1][1], dg::equals(), wi, R,Z,P,time);
    double normne = sqrt(dg::blas2::dot( w3d, sol[0][0]));
    double normni = sqrt(dg::blas2::dot( w3d, sol[0][1]));
    double normwe = sqrt(dg::blas2::dot( w3d, sol[1][0]));
    double normwi = sqrt(dg::blas2::dot( w3d, sol[1][1]));

    dg::blas1::axpby( 1., sol, -1., y);
    std::cout <<"           rel. Error\tNorm: \n"
              <<"    ne:   "<<sqrt(dg::blas2::dot( w3d, y[0][0]))/normne<<"\t"<<normne<<"\n"
              <<"    ni:   "<<sqrt(dg::blas2::dot( w3d, y[0][1]))/normni<<"\t"<<normni<<"\n"
              <<"    we:   "<<sqrt(dg::blas2::dot( w3d, y[1][0]))/normwe<<"\t"<<normwe<<"\n"
              <<"    wi:   "<<sqrt(dg::blas2::dot( w3d, y[1][1]))/normwi<<"\t"<<normwi<<"\n";



    return 0;
}
