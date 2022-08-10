#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/file.h"
#include "feltordiag.h"

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
    err = nc_close( ncid_in);
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    dg::file::string2Json(inputfile, js.asJson(), dg::file::comments::are_forbidden);
    //we only need some parameters from p, not all
    const feltor::Parameters p(js);
    std::cout << js.asJson() <<  std::endl;

    //-------------------Construct grids-------------------------------------//

    const double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    const double Zmin=-p.boxscaleZm*mag.params().a();
    const double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    const double Zmax=p.boxscaleZp*mag.params().a();

    /////////////////////////////////////////////////////////////////////////
    dg::CylindricalGrid3d g3d( Rmin, Rmax, Zmin, Zmax, 0., 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
    std::cout << "Opening file "<<argv[1]<<"\n";
    std::string names[7] = {"chi", "sol", "rhs", "ne", "Ni", "phiH", "phi0"};
    dg::x::HVec transferH = dg::evaluate( dg::zero, g3d);
    for( int i =0; i<7; i++)
    {
        err = nc_inq_varid(ncid, names[i].data(), &dataID);
        err = nc_get_var_double( ncid_in, dataID,
                        transferH.data());
    }
    dg::MultigridCG2d<Geometry, Matrix, Container> multigrid( g3d, p.stages);
    multigrid.set_max_iter( 1e5);
    std::vector<dg::Elliptic3d< Geometry, Matrix, Container> > multi_pol;
    std::vector<dg::Helmholtz3d<Geometry, Matrix, Container> > m_multi_invgammaN;

    return 0;
}
