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
    size_t length;
    err = nc_inq_attlen( ncid_in, NC_GLOBAL, "inputfile", &length);
    std::string inputfile(length, 'x');
    err = nc_get_att_text( ncid_in, NC_GLOBAL, "inputfile", &inputfile[0]);
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json(inputfile, dg::file::comments::are_forbidden);
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

    const double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    const double Zmin=-p.boxscaleZm*mag.params().a();
    const double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    const double Zmax=p.boxscaleZp*mag.params().a();

    /////////////////////////////////////////////////////////////////////////
    dg::CylindricalGrid3d g3d_fine( Rmin, Rmax, Zmin, Zmax, 0., 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);

    std::cout << "Reading file "<<argv[1]<<"\n";
    std::string names[4] = {"chi", "sol", "rhs", "phi0"};
    dg::x::HVec transferH = dg::evaluate( dg::zero, g3d_fine);
    std::map<std::string, dg::x::HVec> vecs;
    for( int i =0; i<4; i++)
    {
        int dataID;
        err = nc_inq_varid(ncid_in, names[i].data(), &dataID);
        err = nc_get_var_double( ncid_in, dataID,
                        transferH.data());
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

    int ncid_out, vecID;
    int dim_ids[3];
    err = nc_create( argv[2], NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err = dg::file::define_dimensions( ncid_out, dim_ids, g3d,
                {"z", "y", "x"});
    std::string out_names [3] = { "sol", "rhs", "guess"};
    std::cout << "Write output "<<argv[2]<<"\n";
    for ( unsigned i=0; i<3; i++)
    {
        err = nc_def_var( ncid_out, out_names[i].data(), NC_DOUBLE, 3,
                    dim_ids, &vecID);
        if( out_names[i] == "guess")
            //dg::file::put_var_double( ncid_out, vecID, g3d, vecs["phi0"]);
            dg::file::put_var_double( ncid_out, vecID, g3d, multi_phi[STAGE]);
        else if( out_names[i] == "sol")
            dg::file::put_var_double( ncid_out, vecID, g3d, multi_sol[STAGE]);
        else if( out_names[i] == "rhs")
            dg::file::put_var_double( ncid_out, vecID, g3d, multi_rhs[STAGE]);
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
    dg::Grid1d g1d_nnz( 0,1 , 1, result.num_entries);
    dg::Grid1d g1d_dimi( 0,1 , 1, result.num_rows+1);
    int dim_nnz_id, dim_dimi_id;
    err = dg::file::define_dimension( ncid_out, &dim_nnz_id, g1d_nnz, "nnz");
    err = dg::file::define_dimension( ncid_out, &dim_dimi_id, g1d_dimi, "dimi");
    err = nc_def_var( ncid_out, "i", NC_INT, 1, &dim_dimi_id, &vecID);
    err = nc_put_var_int( ncid_out, vecID, &result.row_offsets[0]);
    err = nc_def_var( ncid_out, "j", NC_INT, 1, &dim_nnz_id, &vecID);
    err = nc_put_var_int( ncid_out, vecID, &result.column_indices[0]);
    err = nc_def_var( ncid_out, "val", NC_DOUBLE, 1, &dim_nnz_id, &vecID);
    err = nc_put_var_double( ncid_out, vecID, &result.values[0]);
    int num_rows = result.num_rows, num_cols = result.num_cols;
    err = nc_put_att_int( ncid_out, NC_GLOBAL, "ndim", NC_INT, 1,
		    &num_rows);
    err = nc_put_att_int( ncid_out, NC_GLOBAL, "ncol", NC_INT, 1,
		    &num_cols);
    err = nc_close(ncid_out);

    return 0;
}
