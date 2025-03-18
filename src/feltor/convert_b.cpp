#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <cusp/elementwise.h>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/file.h"
#include "feltordiag.h"
#include "common.h"

using Geometry =  dg::x::CylindricalGrid3d;
using Matrix = dg::x::HMatrix;
using Container = dg::x::HVec;

//using IHMatrix = cusp::coo_matrix<int, double, cusp::host_memory>;
using IHMatrix = dg::IHMatrix;

int main( int argc, char* argv[])
{
    if( argc < 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [polarisation.nc] [output.nc] \n";
        return -1;
    }
    std::cout << argv[0] <<" "<<argv[1]<<" -> "<<argv[2]<<std::endl;

    //------------------------open input nc file--------------------------------//
    dg::file::NcFile file_in( argv[1], dg::file::nc_nowrite);
    std::string inputfile = file_in.get_att_as<std::string>( "inputfile");
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json(inputfile,
        dg::file::comments::are_forbidden);
    //we only need some parameters from p, not all
    const feltor::Parameters p(js);
    std::cout << js.toStyledString() <<  std::endl;

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
    dg::CylindricalGrid3d g3d_fine( box.at("Rmin"), box.at("Rmax"), box.at("Zmin"), box.at("Zmax"), 0., 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);

    std::cout << "Reading file "<<argv[1]<<"\n";
    std::string names[4] = {"chi", "sol", "rhs", "phi0"};
    std::map<std::string, dg::x::HVec> vecs;
    for( int i =0; i<4; i++)
    {
        file_in.get_var( names[i], {g3d_fine}, vecs[names[i]]);
    }
    file_in.close();
    dg::MultigridCG2d<Geometry, Matrix, Container> multigrid( g3d_fine, p.stages);
    std::vector<dg::Elliptic3d< Geometry, Matrix, Container> > multi_pol(p.stages);
    std::vector<Container> multi_chi = multigrid.project( vecs["chi"]);
    std::vector<Container> multi_rhs = multigrid.project( vecs["rhs"]);
    std::vector<Container> multi_sol = multigrid.project( vecs["sol"]);
    std::vector<Container> multi_phi = multigrid.project( vecs["phi0"]);
    for( unsigned u=0; u<p.stages; u++)
    {
        multi_pol[u].construct( multigrid.grid(u),
            p.bcxP, p.bcyP, dg::PER,
            p.pol_dir, p.jfactor);

        auto bhat = dg::geo::createEPhi(+1);
        dg::SparseTensor<Container> hh = dg::geo::createProjectionTensor(
            bhat, multigrid.grid(u));
        multi_pol[u].set_chi( hh);
        if(p.curvmode != "true"){
            multi_pol[u].set_compute_in_2d( true);
        }
        multi_pol[u].set_chi( multi_chi[u]);
    }
    // select stage
    unsigned STAGE = p.stages-1;
    Geometry g3d = multigrid.grid(STAGE);
    // Create and write matrix to file
    std::cout << "Create 1. matrices\n";
    IHMatrix leftx = dg::create::dx( g3d, dg::inverse( p.bcxP), dg::inverse(p.pol_dir)).asCuspMatrix();
    dg::blas1::scal( leftx.values, -1.);
    std::cout << "Create 2. matrices\n";
    IHMatrix lefty =  dg::create::dy( g3d, dg::inverse( p.bcyP), dg::inverse(p.pol_dir)).asCuspMatrix();
    dg::blas1::scal( lefty.values, -1.);
    std::cout << "Create 3. matrices\n";
    IHMatrix rightx =  dg::create::dx( g3d, p.bcxP, p.pol_dir).asCuspMatrix();
    IHMatrix righty =  dg::create::dy( g3d, p.bcyP, p.pol_dir).asCuspMatrix();
    std::cout << "Create 4. matrices\n";
    IHMatrix jumpx =  dg::create::jumpX( g3d, p.bcxP).asCuspMatrix();
    IHMatrix jumpy =  dg::create::jumpY( g3d, p.bcyP).asCuspMatrix();
    // Create volume form
    dg::HVec vol3d = dg::tensor::volume( g3d.metric());
    dg::HVec inv_vol3d = vol3d;
    dg::blas1::pointwiseDivide( 1., vol3d, inv_vol3d);
    dg::blas1::pointwiseDot( vol3d, multi_chi[STAGE], vol3d);
    IHMatrix chi_diag = dg::create::diagonal( vol3d);
    IHMatrix inv_vol = dg::create::diagonal( inv_vol3d);
    IHMatrix CX, XX, CY, YY, JJ, result;

    std::cout << "Multiply 1. matrices\n";
    cusp::multiply( chi_diag, rightx, CX);
    std::cout << "Multiply 2. matrices\n";
    cusp::multiply( leftx, CX, XX );
    std::cout << "Multiply 3. matrices\n";
    cusp::multiply( chi_diag, righty, CY);
    std::cout << "Multiply 4. matrices\n";
    cusp::multiply( lefty, CY, YY );
    std::cout << "Add 1. matrices\n";
    cusp::add( jumpx, jumpy, JJ);
    std::cout << "Add 2. matrices\n";
    cusp::add( XX, YY, CX);
    std::cout << "Add 3. matrices\n";
    cusp::add( CX, JJ, XX);
    std::cout << "Multiply 5. matrices\n";
    cusp::multiply( inv_vol, XX, result);
    //std::cout << "Sort\n";
    //result.sort_by_row();
    std::cout << "Done\n";

    //int dim_ids[3];
    dg::file::NcFile file( argv[2], dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"},
        {"long_name", "R-coordinate in Cylindrical system"}},
        g3d.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"},
        {"long_name", "Z-coordinate in Cylindrical system"}},
        g3d.abscissas(1));
    file.defput_dim( "z", {{"axis", "Z"},
        {"long_name", "Phi-coordinate in Cylindrical system"}},
        g3d.abscissas(2));
    std::cout << "Write output "<<argv[2]<<"\n";
    file.defput_var( "sol", {"z","y","x"}, {}, g3d, multi_sol[STAGE]);
    file.defput_var( "rhs", {"z","y","x"}, {}, g3d, multi_rhs[STAGE]);
    file.defput_var( "guess", {"z","y","x"}, {}, g3d, multi_phi[STAGE]);
    std::cout << "Done!\n";
    std::cout << "Compare matrices!\n";
    // Test the matrix if it converges
    Container w3d = dg::create::volume( g3d);
    Container phi0 = multi_phi[STAGE];
    Container rhs0 = multi_rhs[STAGE], rhs1(rhs0);

    dg::blas2::symv( multi_pol[STAGE], phi0, rhs0);
    dg::blas2::symv( result, phi0, rhs1);
    dg::blas1::axpby( 1., rhs0, -1., rhs1);
    double error = dg::blas2::dot( rhs1, w3d, rhs1);
    std::cout<< "Norm rhs "<<sqrt(dg::blas1::dot( multi_rhs[STAGE], multi_rhs[STAGE]))<<"\n";
    std::cout<< "Norm guess "<<sqrt(dg::blas1::dot( multi_phi[STAGE], multi_phi[STAGE]))<<"\n";
    std::cout<< "Norm matrix "<<sqrt(dg::blas1::dot( result.values, result.values))<<"\n";
    std::cout << "Compare solution with elliptic matrix "<<error <<"\n";
    dg::PCG<Container> cg( w3d, 2000);
    rhs0 = multi_rhs[STAGE];
    cg.set_verbose(true);
    // why is result faster than multi_pol?? Maybe because no vectorization...
    unsigned number = cg.solve( result, phi0, rhs0, multi_pol[STAGE].precond(), w3d, p.eps_pol[STAGE], 1, 10);
    std::cout << "CG solver takes "<<number<<" iterations\n";
    // Write out matrix
    file.put_att( {"ndim", unsigned(result.num_rows)});
    file.put_att( {"ncol", unsigned(result.num_cols)});
    file.defput_dim( "i", {}, result.row_offsets);
    file.defput_dim( "j", {}, result.column_indices);
    file.defput_var( "vals", {"j"}, {}, {result.values}, result.values);

    file.close();

    return 0;
}
