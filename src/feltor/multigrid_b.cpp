#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/file.h"
#include "feltordiag.h"

using Geometry =  dg::x::CylindricalGrid3d;
using Matrix = dg::x::DMatrix;
using Container = dg::x::DVec;

int main( int argc, char* argv[])
{
    if( argc < 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [polarisation.nc] \n";
        return -1;
    }
    std::cout << argv[0] <<" -> "<<argv[1]<<std::endl;

    //------------------------open input nc file--------------------------------//
    dg::file::NcFile file_in( argv[1], dg::file::nc_nowrite);
    std::string inputfile = file_in.get_att_as<std::string>( "inputfile");
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json(inputfile,
        dg::file::comments::are_forbidden);
    //we only need some parameters from p, not all
    const feltor::Parameters p(js);
    std::cout << js.asJson() <<  std::endl;

    //-------------------Construct grids-------------------------------------//
    dg::geo::TokamakMagneticField mag;
    try{
        mag = dg::geo::createMagneticField(js["magnetic_field"]["params"]);
    }catch(std::runtime_error& e)
    {
        std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
        std::cerr <<e.what()<<std::endl;
        return -1;
    }

    /////////////////////////////////////////////////////////////////////////
    auto box = common::box( js);
    dg::CylindricalGrid3d g3d( box.at("Rmin"), box.at("Rmax"), box.at("Zmin"), box.at("Zmax"), 0., 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::CylindricalGrid3d g2d( box.at("Rmin"), box.at("Rmax"), box.at("Zmin"), box.at("Zmax"), 0., 2.*M_PI,
    //    p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
        p.n, p.Nx, p.Ny, 1, p.bcxN, p.bcyN, dg::PER);
    std::cout << "Opening file "<<argv[1]<<"\n";
    std::string names[7] = {"chi", "sol", "rhs", "ne", "Ni", "phiH", "phi0"};
    dg::x::HVec transferH = dg::evaluate( dg::zero, g3d);
    dg::x::HVec transferH2d = dg::evaluate( dg::zero, g2d);
    std::map<std::string, dg::x::DVec> vecs;
    dg::x::DVec weights = dg::create::volume( g2d);
    dg::x::HVec weightsH = dg::create::volume( g3d);
    dg::x::HVec phi_sol = dg::create::volume( g3d);
    auto split_sol = dg::split( phi_sol, g3d);

    //for( unsigned k =0; k<p.Nz; k++)
    {

    std::cout << "Plane? \n";
    unsigned k=4;
    std::cin >>k;
    std::cout << "Plane "<<k<<"\n";
    unsigned num_stages = p.stages;
    std::cout << "Num stages\n";
    std::cin >> num_stages;
    std::cout << " "<<num_stages<<"\n";
    std::cout << "Eps_pol  [1e-6,1e-6,1e-6]?\n";
    std::vector<double> eps_pol(num_stages);
    for( unsigned u=0; u<num_stages; u++)
        std::cin >> eps_pol[u];
    std::cout << "Eps: ";
    for( unsigned u=0; u<num_stages; u++)
        std::cout << eps_pol[u]<<", ";
    std::cout << "\n";
    std::cout << "Eps_gamma  [1e-7,1e-7,1e-7]?\n";
    std::vector<double> eps_gamma(num_stages);
    for( unsigned u=0; u<num_stages; u++)
        std::cin >> eps_gamma[u];
    for( unsigned u=0; u<num_stages; u++)
        std::cout << eps_gamma[u]<<", ";
    std::cout << "\n";
    std::cout << "Direction (forward)\n";
    std::string direction;
    std::cin >> direction;
    dg::direction dir = dg::str2direction( direction);
    std::cout <<"Jfactor (1)\n";
    double jfactor = 1.;
    std::cin >> jfactor;
    for( int i =0; i<7; i++)
    {
        file_in.get_var( names[i], {g3d}, transferH);
        auto split_view = dg::split( transferH, g3d);
        dg::assign( split_view[k], vecs[names[i]]);
        //dg::assign( transferH, vecs[names[i]]);
    }
    for( auto pair : vecs)
    {
        double norm = dg::blas2::dot( weights, pair.second);
        std::cout << pair.first << " "<<sqrt(norm)<<"\n";
    }
    dg::MultigridCG2d<Geometry, Matrix, Container> multigrid( g2d, num_stages);
    multigrid.set_max_iter( 1e5);
    std::vector<dg::Elliptic3d< Geometry, Matrix, Container> > multi_pol(num_stages);
    std::vector<dg::Helmholtz3d<Geometry, Matrix, Container> > multi_invgammaN(num_stages);
    std::vector<Container> multi_chi = multigrid.project( vecs["chi"]);
    for( unsigned u=0; u<num_stages; u++)
    {

        multi_pol[u].construct( multigrid.grid(u),
            p.bcxP, p.bcyP, dg::PER,
            dir, jfactor);
        multi_invgammaN[u] = { -0.5*p.tau[1]*p.mu[1],
                {multigrid.grid(u), p.bcxN, p.bcyN, dg::PER, dir}};

        auto bhat = dg::geo::createEPhi(+1);
        dg::SparseTensor<Container> hh = dg::geo::createProjectionTensor(
            bhat, multigrid.grid(u));
        multi_pol[u].set_chi( hh);
        multi_invgammaN[u].matrix().set_chi( hh);
        if(p.curvmode != "true"){
            multi_pol[u].set_compute_in_2d( true);
            multi_invgammaN[u].matrix().set_compute_in_2d( true);
        }
    }
    dg::x::DVec temp0(vecs["Ni"]), temp1(temp0);
    dg::blas1::transform( vecs["Ni"], temp1, dg::PLUS<double>(-p.nbc));
    multigrid.set_benchmark( true, "Gamma N     ");
    dg::blas1::copy( temp1, temp0);
    std::vector<unsigned> numberG = multigrid.solve(
            multi_invgammaN, temp0, temp1, eps_gamma);
    dg::blas1::transform( vecs["ne"], temp1, dg::PLUS<double>(-p.nbc));
    dg::blas1::axpby( -1., temp1, 1., temp0, temp0);
    dg::blas1::axpby( -1., vecs["rhs"], 1., temp0, temp0);
    double error = dg::blas2::dot( temp0, weights, temp0);
    double norm = dg::blas2::dot( vecs["rhs"], weights, vecs["rhs"]);
    std::cout << "Norm error Gamma N "<<sqrt(error/norm)<<"\n";

    multigrid.project( vecs["chi"], multi_chi);
    for( unsigned u=0; u<num_stages; u++)
        multi_pol[u].set_chi( multi_chi[u]);
    multigrid.set_benchmark( true, "Polarisation");
    dg::x::DVec phi = vecs["phi0"];
    std::vector<unsigned> number = multigrid.solve(
        multi_pol, phi, vecs["rhs"], eps_pol);
    dg::blas1::axpby( -1., phi, 1., vecs["sol"], temp0);
    error = dg::blas2::dot( temp0, weights, temp0);
    norm = dg::blas2::dot( vecs["sol"], weights, vecs["sol"]);
    std::cout << "Norm error Inv Phi "<<sqrt(error/norm)<<"\n";
    dg::assign( phi, transferH2d);
    dg::blas1::copy( transferH2d, split_sol[k]);
    }
    file_in.close();

    dg::file::NcFile file( "pol_out.nc", dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"},
        {"long_name", "R-coordinate in Cylindrical system"}},
        g2d.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"},
        {"long_name", "Z-coordinate in Cylindrical system"}},
        g2d.abscissas(1));
    file.defput_dim( "z", {{"axis", "Z"},
        {"long_name", "Phi-coordinate in Cylindrical system"}},
        g2d.abscissas(2));
    file.defput_var( "phi", {"z", "y", "x"}, {}, {g2d}, transferH2d);
    file.close();

    return 0;
}
