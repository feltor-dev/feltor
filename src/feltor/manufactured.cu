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

int main( int argc, char* argv[])
{
    dg::file::WrappedJsonValue js( dg::file::error::is_throw);
    std::string inputfile = argc == 1 ? "input.json" : argv[1];
    dg::file::file2Json( inputfile, js.asJson(), dg::file::comments::are_discarded);

    const feltor::Parameters p( js);
    std::cout << js.asJson() <<std::endl;
    dg::geo::TokamakMagneticField mag = dg::geo::createMagneticField( js["params"]);

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
    std::cout << "# Construct rhs" << std::endl;
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::x::DVec> feltor( grid, p, mag);

    feltor::manufactured::Ne ne{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp_n,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ni ni{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp_n,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ue ue{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp_n,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Ui ui{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                 p.beta,p.nu_perp_n,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Phie phie{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                     p.beta,p.nu_perp_n,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::Phii phii{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                     p.beta,p.nu_perp_n,p.nu_parallel[0],p.nu_parallel[1]};
    feltor::manufactured::A aa{ p.mu[0],p.mu[1],p.tau[0],p.tau[1],p.eta,
                                p.beta,p.nu_perp_n,p.nu_parallel[0],p.nu_parallel[1]};

    dg::x::DVec R = dg::pullback( dg::cooX3d, grid);
    dg::x::DVec Z = dg::pullback( dg::cooY3d, grid);
    dg::x::DVec P = dg::pullback( dg::cooZ3d, grid);
    std::array<dg::x::DVec,2> phi{R,R}, sol_phi{phi};
    std::array<std::array<dg::x::DVec,2>,2> y0{phi,phi}, sol{y0};
    dg::x::DVec apar{R}, sol_apar{apar};
    dg::blas1::evaluate( y0[0][0], dg::equals(), ne, R,Z,P,0);
    dg::blas1::evaluate( y0[0][1], dg::equals(), ni, R,Z,P,0);
    dg::blas1::evaluate( y0[1][0], dg::equals(), ue, R,Z,P,0);
    dg::blas1::evaluate( y0[1][1], dg::equals(), ui, R,Z,P,0);
    dg::blas1::evaluate( apar, dg::equals(), aa, R,Z,P,0);
    dg::blas1::plus(y0[0][0],-1); //ne-1
    dg::blas1::plus(y0[0][1],-1); //Ni-1
    dg::blas1::axpby(1./p.mu[0], apar, 1., y0[1][0]); //we=ue+1/mA
    dg::blas1::axpby(1./p.mu[1], apar, 1., y0[1][1]); //Wi=Ui+1/mA

    dg::ExplicitMultistep< std::array<std::array<dg::x::DVec,2>,2 > > mp(p.tableau, y0);
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
    const std::array<std::array<dg::x::DVec,2>,2>& num = feltor.fields();
    const std::array<dg::x::DVec,2>& num_phi = feltor.potentials();
    const dg::x::DVec& num_apar = feltor.aparallel();
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
    double errone = sqrt(dg::blas2::dot( w3d, sol[0][0]));
    double erroni = sqrt(dg::blas2::dot( w3d, sol[0][1]));
    double erroue = sqrt(dg::blas2::dot( w3d, sol[1][0]));
    double erroui = sqrt(dg::blas2::dot( w3d, sol[1][1]));
    double errophie = sqrt(dg::blas2::dot( w3d, sol_phi[0]));
    double errophii = sqrt(dg::blas2::dot( w3d, sol_phi[1]));
    double erroapar = sqrt(dg::blas2::dot( w3d, sol_apar));
    std::cout<<std::scientific;
    std::cout <<"# Computed with\n"
              <<"n:  "<<p.n<<"\n"
              <<"Nx: "<<p.Nx<<"\n"
              <<"Ny: "<<p.Ny<<"\n"
              <<"Nz: "<<p.Nz<<"\n"
              <<"Time: "<<time<<"\n";
    std::cout <<"error:\n"
              <<"    ne:   "<<errone/normne<<"\n"
              <<"    ni:   "<<erroni/normni<<"\n"
              <<"    ue:   "<<erroue/normue<<"\n"
              <<"    ui:   "<<erroui/normui<<"\n"
              <<"    phie: "<<errophie/normphie<<"\n"
              <<"    phii: "<<errophii/normphii<<"\n"
              <<"    apar: "<<erroapar/normapar<<"\n";
    std::cout << "norm:\n"
              <<"    ne:   "<<normne<<"\n"
              <<"    ni:   "<<normni<<"\n"
              <<"    ue:   "<<normue<<"\n"
              <<"    ui:   "<<normui<<"\n"
              <<"    phie: "<<normphie<<"\n"
              <<"    phii: "<<normphii<<"\n"
              <<"    apar: "<<normapar<<"\n";


    return 0;

}
