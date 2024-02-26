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
    dg::file::NC_Error_Handle err;
    int ncid_in;
    err = nc_open( argv[1], NC_NOWRITE, &ncid_in); //open 3d file
    dg::file::WrappedJsonValue jsin = dg::file::nc_attrs2json( ncid_in, NC_GLOBAL);
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json(jsin["inputfile"].asString(),
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
    dg::x::HVec transferH = dg::evaluate( dg::zero, g3d_fine);
    std::map<std::string, dg::x::HVec> vecs;
    dg::file::Reader<dg::x::CylindricalGrid3d> read3d( ncid_in, g3d_fine, {"z","y","x"});
    for( int i =0; i<4; i++)
    {
        read3d.get( names[i], transferH);
        dg::assign( transferH, vecs[names[i]]);
    }
    nc_close(ncid_in);
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

    int ncid_out;//, vecID;
    //int dim_ids[3];
    err = nc_create( argv[2], NC_NETCDF4|NC_CLOBBER, &ncid_out);
    dg::file::Writer<dg::x::CylindricalGrid3d> writer( ncid_out, g3d,
                {"z", "y", "x"});
    std::string out_names [3] = { "sol", "rhs", "guess"};
    std::cout << "Write output "<<argv[2]<<"\n";
    for ( unsigned i=0; i<3; i++)
    {
        writer.def( out_names[i], {});
        if( out_names[i] == "guess")
            //writer.put( out_names[i], vecs["phi0"]);
            writer.put( out_names[i], multi_phi[STAGE]);
        else if( out_names[i] == "sol")
            writer.put( out_names[i], multi_sol[STAGE]);
        else if( out_names[i] == "rhs")
            writer.put( out_names[i], multi_rhs[STAGE]);
    }
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
    dg::RealGrid1d<int> g1d_nnz( 0,result.num_entries , 1, result.num_entries);
    dg::file::Writer<dg::RealGrid1d<int>>( ncid_out, g1d_nnz,
        {"nnz"}).def_and_put( "j", {}, result.column_indices);

    dg::RealGrid1d<int> g1d_dimi( 0,result.num_rows+1 , 1, result.num_rows+1);
    dg::file::Writer<dg::RealGrid1d<int>>( ncid_out, g1d_dimi,
        {"dimi"}).def_and_put( "i", {}, result.row_offsets);

    dg::RealGrid1d<double> g1d_val( 0,result.num_entries , 1, result.num_entries);
    dg::file::Writer<dg::RealGrid1d<double>>( ncid_out, g1d_val,
        {"dimv"}).def_and_put( "val", {}, result.values);

    dg::file::JsonType att;
    att["ndim"] = result.num_rows;
    att["ncol"] = result.num_cols;
    dg::file::json2nc_attrs( att, ncid_out, NC_GLOBAL);

    err = nc_close(ncid_out);

    return 0;
}
