#include <iostream>
#include <iomanip>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/json_utilities.h"

#include "parameters.h"
#define DG_MANUFACTURED
//Changed by Makefile
// #define FELTORPARALLEL 1
// #define FELTORPERP 1

#include "manufactured.h"
#include "feltor.h"
void abort_program(int code = -1){
#ifdef WITH_MPI
    MPI_Abort(MPI_COMM_WORLD, code);
#endif //WITH_MPI
    exit( code);
}

// ATTENTION: add parallel diffusion in n needed in manufactured solution
int main( int argc, char* argv[])
{
    dg::file::WrappedJsonValue js( dg::file::error::is_throw);
    std::string inputfile = argc == 1 ? "input.json" : argv[1];
    dg::file::file2Json( inputfile, js.asJson(), dg::file::comments::are_discarded);

    const feltor::Parameters p( js);
    DG_RANK0 std::cout << js.asJson() <<std::endl;
    dg::geo::TokamakMagneticField mag = dg::geo::createMagneticField(
            js["magnetic_field"]["params"]);

    double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    double Zmin=-p.boxscaleZm*mag.params().a();
    double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    double Zmax=p.boxscaleZp*mag.params().a();
    dg::x::CylindricalGrid3d grid( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );
    dg::x::DVec w3d = dg::create::volume( grid);

    //create RHS
    DG_RANK0 std::cout << "# Construct rhs" << std::endl;
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix,
        dg::x::DVec> feltor( grid, p, mag, js);

    feltor::manufactured::Ne ne{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
        p.beta,p.nu_perp_n,p.nu_parallel_u[0],p.nu_parallel_u[1]};
    feltor::manufactured::Ni ni{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
        p.beta,p.nu_perp_n,p.nu_parallel_u[0],p.nu_parallel_u[1]};
    feltor::manufactured::Ue ue{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
        p.beta,p.nu_perp_n,p.nu_parallel_u[0],p.nu_parallel_u[1]};
    feltor::manufactured::Ui ui{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
        p.beta,p.nu_perp_n,p.nu_parallel_u[0],p.nu_parallel_u[1]};
    feltor::manufactured::Phie phie{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
        p.beta,p.nu_perp_n,p.nu_parallel_u[0],p.nu_parallel_u[1]};
    feltor::manufactured::Phii phii{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
        p.beta,p.nu_perp_n,p.nu_parallel_u[0],p.nu_parallel_u[1]};
    feltor::manufactured::A aa{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
        p.beta,p.nu_perp_n,p.nu_parallel_u[0],p.nu_parallel_u[1]};

    dg::x::DVec R = dg::pullback( dg::cooX3d, grid);
    dg::x::DVec Z = dg::pullback( dg::cooY3d, grid);
    dg::x::DVec P = dg::pullback( dg::cooZ3d, grid);
    dg::x::DVec PST = dg::pullback( dg::cooZ3d, grid);
    dg::blas1::plus(PST, grid.hz()/2.);

    std::array<dg::x::DVec,2> phi{R,R}, sol_phi{phi};
    std::array<std::array<dg::x::DVec,2>,2> y0{phi,phi}, sol{y0};
    dg::x::DVec apar{R}, sol_apar{apar};
    dg::blas1::evaluate( y0[0][0], dg::equals(), ne, R,Z,P,0);
    dg::blas1::evaluate( y0[0][1], dg::equals(), ni, R,Z,P,0);
    dg::blas1::evaluate( y0[1][0], dg::equals(), ue, R,Z,PST,0);
    dg::blas1::evaluate( y0[1][1], dg::equals(), ui, R,Z,PST,0);
    dg::blas1::evaluate( apar, dg::equals(), aa, R,Z,PST,0);
    dg::blas1::axpby(1./p.mu[0], apar, 1., y0[1][0]); //we=ue+1/mA
    dg::blas1::axpby(1./p.mu[1], apar, 1., y0[1][1]); //Wi=Ui+1/mA

    dg::ExplicitMultistep< std::array<std::array<dg::x::DVec,2>,2 > > multistep;
    dg::Adaptive< dg::ERKStep< std::array<std::array<dg::x::DVec,2>,2>>> adapt;
    double time = 0;
    double rtol = 0., atol = 0., dt = 0.;
    if( p.timestepper == "multistep")
    {
        multistep.construct( p.tableau, y0);
        dt = js[ "timestepper"]["dt"].asDouble( 0.01);
    }
    else if (p.timestepper == "adaptive")
    {
        adapt.construct( p.tableau, y0);
        rtol = js[ "timestepper"][ "rtol"].asDouble( 1e-7);
        atol = js[ "timestepper"][ "atol"].asDouble( 1e-10);
        dt = 1e-4; //that should be a small enough initial guess
    }
    else
    {
        DG_RANK0 std::cerr<<"Error: Unrecognized timestepper: '"<<p.timestepper<<"'! Exit now!";
        abort_program();
        return -1;
    }
    DG_RANK0 std::cout << "Done!\n";
    if( p.timestepper == "multistep")
        multistep.init( feltor, time, y0, dt);
    unsigned maxout = js["output"].get( "maxout", 0).asUInt();
    for( unsigned j=0; j<p.itstp*maxout; j++)
    {
        for( unsigned i=0; i<p.inner_loop; i++)
        {
            try{
                if( p.timestepper == "adaptive")
                    adapt.step( feltor, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, rtol, atol);
                if( p.timestepper == "multistep")
                    multistep.step( feltor, time, y0);
            }
            catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                return -1;
            }
        }
        DG_RANK0 std::cout << "#Time "<<time<<std::endl;
    }
    //now compare stuff
    dg::blas1::evaluate( sol[0][0], dg::equals(), ne, R,Z,P,time);
    dg::blas1::evaluate( sol[0][1], dg::equals(), ni, R,Z,P,time);
    dg::blas1::evaluate( sol[1][0], dg::equals(), ue, R,Z,P,time);
    dg::blas1::evaluate( sol[1][1], dg::equals(), ui, R,Z,P,time);
    dg::blas1::evaluate( sol_apar, dg::equals(), aa, R,Z,P,time);
    dg::blas1::evaluate( sol_phi[0], dg::equals(),phie,R,Z,P,time);
    dg::blas1::evaluate( sol_phi[1], dg::equals(),phii,R,Z,P,time);
    double normne = sqrt(dg::blas2::dot( w3d, sol[0][0]));
    double normni = sqrt(dg::blas2::dot( w3d, sol[0][1]));
    double normue = sqrt(dg::blas2::dot( w3d, sol[1][0]));
    double normui = sqrt(dg::blas2::dot( w3d, sol[1][1]));
    double normphie = sqrt(dg::blas2::dot( w3d, sol_phi[0]));
    double normphii = sqrt(dg::blas2::dot( w3d, sol_phi[1]));
    double normapar = sqrt(dg::blas2::dot( w3d, sol_apar));
    dg::blas1::axpby( 1., feltor.density(0), -1.,sol[0][0]);
    dg::blas1::axpby( 1., feltor.density(1), -1.,sol[0][1]);
    dg::blas1::axpby( 1., feltor.velocity(0), -1.,sol[1][0]);
    dg::blas1::axpby( 1., feltor.velocity(1), -1.,sol[1][1]);
    dg::blas1::axpby( 1., feltor.potential(0), -1.,sol_phi[0]);
    dg::blas1::axpby( 1., feltor.potential(1), -1.,sol_phi[1]);
    dg::blas1::axpby( 1., feltor.aparallel(), -1.,sol_apar);
    double errone = sqrt(dg::blas2::dot( w3d, sol[0][0]));
    double erroni = sqrt(dg::blas2::dot( w3d, sol[0][1]));
    double erroue = sqrt(dg::blas2::dot( w3d, sol[1][0]));
    double erroui = sqrt(dg::blas2::dot( w3d, sol[1][1]));
    double errophie = sqrt(dg::blas2::dot( w3d, sol_phi[0]));
    double errophii = sqrt(dg::blas2::dot( w3d, sol_phi[1]));
    double erroapar = sqrt(dg::blas2::dot( w3d, sol_apar));
    DG_RANK0 std::cout<<std::scientific;
    DG_RANK0 std::cout <<"# Computed with\n"
              <<"n:  "<<p.n<<"\n"
              <<"Nx: "<<p.Nx<<"\n"
              <<"Ny: "<<p.Ny<<"\n"
              <<"Nz: "<<p.Nz<<"\n"
              <<"Time: "<<time<<"\n";
    DG_RANK0 std::cout <<"error:\n"
              <<"    ne:   "<<errone<<"\n"
              <<"    ni:   "<<erroni<<"\n"
              <<"    ue:   "<<erroue<<"\n"
              <<"    ui:   "<<erroui<<"\n"
              <<"    phie: "<<errophie<<"\n"
              <<"    phii: "<<errophii<<"\n"
              <<"    apar: "<<erroapar<<"\n";
    DG_RANK0 std::cout << "norm:\n"
              <<"    ne:   "<<normne<<"\n"
              <<"    ni:   "<<normni<<"\n"
              <<"    ue:   "<<normue<<"\n"
              <<"    ui:   "<<normui<<"\n"
              <<"    phie: "<<normphie<<"\n"
              <<"    phii: "<<normphii<<"\n"
              <<"    apar: "<<normapar<<"\n";


    return 0;

}
