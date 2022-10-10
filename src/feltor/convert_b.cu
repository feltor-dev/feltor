#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <cusp/elementwise.h>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/file.h"
#include "feltordiag.h"

using Geometry =  dg::x::CylindricalGrid3d;
using Matrix = dg::x::DMatrix;
using Container = dg::x::DVec;

using IHMatrix = cusp::coo_matrix<int, double, cusp::host_memory>;

int main( int argc, char* argv[])
{
    if( argc < 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [polarisation.nc] \n";
        return -1;
    }
    std::cout << argv[0] <<" -> "<<argv[1]<<std::endl;

    //------------------------open input nc file--------------------------------//
    dg::file::NC_Error_Handle err;
    int ncid_in;
    err = nc_open( argv[1], NC_NOWRITE, &ncid_in); //open 3d file
    size_t length;
    err = nc_inq_attlen( ncid_in, NC_GLOBAL, "inputfile", &length);
    std::string inputfile(length, 'x');
    err = nc_get_att_text( ncid_in, NC_GLOBAL, "inputfile", &inputfile[0]);
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    dg::file::string2Json(inputfile, js.asJson(), dg::file::comments::are_forbidden);
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

    const double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    const double Zmin=-p.boxscaleZm*mag.params().a();
    const double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    const double Zmax=p.boxscaleZp*mag.params().a();

    /////////////////////////////////////////////////////////////////////////
    dg::CylindricalGrid3d g3d( Rmin, Rmax, Zmin, Zmax, 0., 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
    std::cout << "Reading file "<<argv[1]<<"\n";
    std::string names[4] = {"chi", "sol", "rhs", "phi0"};
    dg::x::HVec transferH = dg::evaluate( dg::zero, g3d);
    std::map<std::string, dg::x::HVec> vecs;
    for( int i =0; i<3; i++)
    {
        int dataID;
        err = nc_inq_varid(ncid_in, names[i].data(), &dataID);
        err = nc_get_var_double( ncid_in, dataID,
                        transferH.data());
        dg::assign( transferH, vecs[names[i]]);
    }
    nc_close(ncid_in);
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
    dg::HVec vol3d = dg::create::volume( g3d);
    dg::HVec inv_vol3d = dg::create::inv_volume( g3d);
    dg::blas1::pointwiseDot( vol3d, vecs["chi"], vol3d);
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
    std::cout << "Sort\n";
    result.sort_by_row_and_column();
    std::cout << "Done\n";

    int ncid_out, vecID;
    int dim_ids[3];
    err = nc_create( "pol_out.nc", NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err = dg::file::define_dimensions( ncid_out, dim_ids, g3d,
                {"z", "y", "x"});
    std::string out_names [3] = { "sol", "rhs", "phi0"};
    for ( unsigned i=0; i<3; i++)
    {
        err = nc_def_var( ncid_out, out_names[i].data(), NC_DOUBLE, 3,
                    dim_ids, &vecID);
        dg::file::put_var_double( ncid_out, vecID, g3d, vecs[out_names[i]]);
    }
    // Write out matrix
    dg::Grid1d g1d( 0,1 , 1, result.num_entries);
    int dim_matrix_id;
    err = dg::file::define_dimension( ncid_out, &dim_matrix_id, g1d);
    err = nc_def_var( ncid_out, "row_indices", NC_INT, 1, &dim_matrix_id, &vecID);
    err = nc_put_var_int( ncid_out, vecID, &result.row_indices[0]);
    err = nc_def_var( ncid_out, "column_indices", NC_INT, 1, &dim_matrix_id, &vecID);
    err = nc_put_var_int( ncid_out, vecID, &result.column_indices[0]);
    err = nc_def_var( ncid_out, "values", NC_DOUBLE, 1, &dim_matrix_id, &vecID);
    err = nc_put_var_double( ncid_out, vecID, &result.values[0]);
    err = nc_close(ncid_out);

    return 0;
}
