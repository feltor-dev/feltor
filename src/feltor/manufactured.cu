#include <iostream>
#include <iomanip>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/json_utilities.h"

#include "parameters.h"
#define DG_MANUFACTURED
//Change here to selectively test parallel and perp parts
#define FELTORPARALLEL 1
#define FELTORPERP 0

#include "manufactured.h"
#include "feltor.h"
#include "implicit.h"

int main( int argc, char* argv[])
{
    Json::Value js, gs;
    if( argc == 1)
        dg::file::file2Json( "input.json", js, dg::file::comments::are_forbidden);
    else if( argc == 2)
        dg::file::file2Json( argv[1], js, dg::file::comments::are_forbidden);
    else
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile]\n";
        return -1;
    }
    const feltor::Parameters p( js, dg::file::error::is_throw);
    std::cout << js <<std::endl;
    const double R_0 = 10;
    const double I_0 = 20; //q factor at r=1 is I_0/R_0
    const double a  = 1; //small radius
    dg::CylindricalGrid3d grid( R_0-a, R_0+a, -a, a, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::DVec w3d = dg::create::volume( grid);

    //create RHS
    std::cout << "Initialize explicit" << std::endl;
    dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> feltor( grid, p, mag);
    //std::cout << "Initialize implicit" << std::endl;
    //feltor::Implicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec > im( grid, p, mag);

    feltor::manufactured::Ne ne{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ni ni{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ue ue{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ui ui{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Phie phie{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                     p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Phii phii{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                     p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::A aa{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp,p.nu_parallel[0],p.nu_parallel[1]};

    dg::DVec R = dg::pullback( dg::cooX3d, grid);
    dg::DVec Z = dg::pullback( dg::cooY3d, grid);
    dg::DVec P = dg::pullback( dg::cooZ3d, grid);
    std::array<dg::DVec,2> phi{R,R}, sol_phi{phi};
    std::array<std::array<dg::DVec,2>,2> y0{phi,phi}, sol{y0};
    dg::DVec apar{R}, sol_apar{apar};
    dg::blas1::evaluate( y0[0][0], dg::equals(), ne, R,Z,P,0);
    dg::blas1::evaluate( y0[0][1], dg::equals(), ni, R,Z,P,0);
    dg::blas1::evaluate( y0[1][0], dg::equals(), ue, R,Z,P,0);
    dg::blas1::evaluate( y0[1][1], dg::equals(), ui, R,Z,P,0);
    dg::blas1::evaluate( apar, dg::equals(), aa, R,Z,P,0);
    dg::blas1::plus(y0[0][0],-1); //ne-1
    dg::blas1::plus(y0[0][1],-1); //Ni-1
    dg::blas1::axpby(1./p.mu[0], apar, 1., y0[1][0]); //we=ue+1/mA
    dg::blas1::axpby(1./p.mu[1], apar, 1., y0[1][1]); //Wi=Ui+1/mA

    //Adaptive solver
    //dg::Adaptive< dg::ARKStep<std::array<std::array<dg::DVec,2>,2>> > adaptive(
    //    "ARK-4-2-3", y0, y0[0][0].size(), p.eps_time);
    //Multistep solver
    //dg::Karniadakis< std::array<std::array<dg::DVec,2>,2 >,
    //    feltor::FeltorSpecialSolver<
    //        dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec>
    //    > karniadakis( grid, p, mag);
    dg::ExplicitMultistep< std::array<std::array<dg::DVec,2>,2 > > mp("TVB-3-3", y0);
    double time = 0, TMAX = 0.1;
    mp.init( feltor, time, y0, p.dt);
    while( time < TMAX)
    {
        try{
            mp.step( feltor, time, y0);
        }
        catch( dg::Fail& fail) {
            std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
            std::cerr << "Does Simulation respect CFL condition?\n";
            return -1;
        }
        std::cout << "#Time "<<time<<std::endl;
    }
    //now compare stuff
    dg::blas1::evaluate( sol[0][0], dg::equals(), ne, R,Z,P,time);
    dg::blas1::evaluate( sol[0][1], dg::equals(), ni, R,Z,P,time);
    dg::blas1::evaluate( sol[1][0], dg::equals(), ue, R,Z,P,time);
    dg::blas1::evaluate( sol[1][1], dg::equals(), ui, R,Z,P,time);
    dg::blas1::evaluate( sol_apar, dg::equals(), aa, R,Z,P,time);
    dg::blas1::evaluate( sol_phi[0], dg::equals(),phie,R,Z,P,time);
    dg::blas1::evaluate( sol_phi[1], dg::equals(),phii,R,Z,P,time);
    dg::blas1::plus(sol[0][0],-1); //ne-1
    dg::blas1::plus(sol[0][1],-1); //Ni-1
    const std::array<std::array<dg::DVec,2>,2>& num = feltor.fields();
    const std::array<dg::DVec,2>& num_phi = feltor.potentials();
    const dg::DVec& num_apar = feltor.induction();
    double normne = sqrt(dg::blas2::dot( w3d, sol[0][0]));
    double normni = sqrt(dg::blas2::dot( w3d, sol[0][1]));
    double normue = sqrt(dg::blas2::dot( w3d, sol[1][0]));
    double normui = sqrt(dg::blas2::dot( w3d, sol[1][1]));
    double normphie = sqrt(dg::blas2::dot( w3d, sol_phi[0]));
    double normphii = sqrt(dg::blas2::dot( w3d, sol_phi[1]));
    double normapar = sqrt(dg::blas2::dot( w3d, sol_apar));
    dg::blas1::axpby( 1., y0[0][0], -1.,sol[0][0]);
    dg::blas1::axpby( 1., y0[0][1], -1.,sol[0][1]);
    dg::blas1::axpby( 1., num[1][0], -1.,sol[1][0]);
    dg::blas1::axpby( 1., num[1][1], -1.,sol[1][1]);
    dg::blas1::axpby( 1., num_phi[0], -1.,sol_phi[0]);
    dg::blas1::axpby( 1., num_phi[1], -1.,sol_phi[1]);
    dg::blas1::axpby( 1., num_apar, -1.,sol_apar);
    std::cout<<std::scientific;
    std::cout <<"           rel. Error\tNorm: \n"
              //<<"    Time: "<<time<<"\n"
              <<"    ne:   "<<sqrt(dg::blas2::dot( w3d, sol[0][0]))/normne<<"\t"<<normne<<"\n"
              <<"    ni:   "<<sqrt(dg::blas2::dot( w3d, sol[0][1]))/normni<<"\t"<<normni<<"\n"
              <<"    ue:   "<<sqrt(dg::blas2::dot( w3d, sol[1][0]))/normue<<"\t"<<normue<<"\n"
              <<"    ui:   "<<sqrt(dg::blas2::dot( w3d, sol[1][1]))/normui<<"\t"<<normui<<"\n"
              <<"    phie: "<<sqrt(dg::blas2::dot( w3d,sol_phi[0]))/normphie<<"\t"<<normphie<<"\n"
              <<"    phii: "<<sqrt(dg::blas2::dot( w3d,sol_phi[1]))/normphii<<"\t"<<normphii<<"\n"
              <<"    apar: "<<sqrt(dg::blas2::dot( w3d,sol_apar))/normapar<<"\t"<<normapar<<"\n";

    return 0;

}
