#include <iostream>
#include <netcdf>
  
#include "json/json.h"
#include "dg/file/file.h"
#include "dg/geometries/geometries.h"
#include "dg/algorithm.h"
#include "ds.h"
#include "nablas.h"


int main()
{
    unsigned n=1, Nx=200, Ny=400, Nz=1;
    dg::geo::TokamakMagneticField mag;
    dg::file::WrappedJsonValue js;
    dg::file::file2Json( "/Users/raulgerru/github_FELTOR/magneticfielddb/data/COMPASS/compass_1X.json", js.asJson(),dg::file::comments::are_discarded, dg::file::error::is_throw);
    mag=dg::geo::createMagneticField(js);
    const double Rmin=mag.R0()-1.5*mag.params().a();
    const double Zmin=-2.0*mag.params().a();
    const double Rmax=mag.R0()+1.5*mag.params().a();
    const double Zmax=2.0*mag.params().a();
    
    std::cout <<"R from "<<Rmin<<" to "<<Rmax<<std::endl;
    std::cout <<"Z from "<<Zmin<<" to "<<Zmax<<std::endl;
    dg::x::CylindricalGrid3d grid( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
    n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    dg::geo::Nablas<dg::x::CylindricalGrid3d, dg::x::DVec, dg::x::DMatrix> nabla(grid, mag, false);
    std::array <dg::HVec, 2> gradPsip, gradPsip_nabla, gradgradPsip;
    dg::HVec Psip, deltaPsip, deltaPsip_nabla, gradPsip_2, gradPsip_2_nabla;
    Psip= dg::evaluate(mag.psip(),grid);
    gradPsip_2= dg::evaluate(dg::zero, grid);
    //DEFINITION OF SOLUTIONS
    gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
    gradPsip_nabla[0]=Psip;
    gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
    gradPsip_nabla[1]=Psip;
    deltaPsip_nabla=Psip;
    gradPsip_2_nabla=Psip;
    deltaPsip=dg::evaluate(dg::geo::LaplacePsip(mag), grid);
    dg::blas1::pointwiseDot(1.0, gradPsip[0], gradPsip[0], 1.0, gradPsip[1], gradPsip[1], 1.0, gradPsip_2);
    
    
    nabla.grad_perp_f(Psip, gradPsip_nabla[0], gradPsip_nabla[1]);
    nabla.v_dot_nabla_f(gradPsip[0], gradPsip[1], Psip, gradPsip_2_nabla);
    nabla.div(gradPsip[0], gradPsip[1], deltaPsip_nabla);
    std::cout <<"After div"<<std::endl;
    
    
     //dg::aRealGeometry2d<double> g2d_out_ptr= grid.perp_grid();
   
    int ncid;
    nc_create("output_nabla_t.nc",NC_NETCDF4|NC_NOCLOBBER, &ncid);
    
    int dim_ids[4];
    size_t start[] = {0, 0};
    size_t count[] = {n*Ny, n*Nx};
    
    dg::file::define_dimensions( ncid, dim_ids, grid, {"z","Z", "R"});
    std::map<std::string, int> ids;
    
    ids["Psip"]=0;
    nc_def_var( ncid, "Psip", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Psip"));
    nc_put_vara_double( ncid, ids.at("Psip"),start, count, Psip.data());
    
    ids["Orig_GradPsipR"]=0;
    nc_def_var( ncid, "Orig_GradPsipR", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Orig_GradPsipR"));
    nc_put_vara_double( ncid, ids.at("Orig_GradPsipR"),start, count, gradPsip[0].data());
    
    ids["Orig_GradPsipZ"]=0;
    nc_def_var( ncid, "Orig_GradPsipZ", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Orig_GradPsipZ"));
    nc_put_vara_double( ncid, ids.at("Orig_GradPsipZ"),start, count, gradPsip[1].data());
    
    ids["Orig_GradPsip_2"]=0;
    nc_def_var( ncid, "Orig_GradPsip_2", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Orig_GradPsip_2"));
    nc_put_vara_double( ncid, ids.at("Orig_GradPsip_2"),start, count, gradPsip_2.data());
    
    ids["Orig_LaplacePsip"]=0;
    nc_def_var( ncid, "Orig_LaplacePsip", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Orig_LaplacePsip"));
    nc_put_vara_double( ncid, ids.at("Orig_LaplacePsip"),start, count, deltaPsip.data());
    
    ids["Nabla_GradPsipR"]=0;
    nc_def_var( ncid, "Nabla_GradPsipR", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Nabla_GradPsipR"));
    nc_put_vara_double( ncid, ids.at("Nabla_GradPsipR"),start, count, gradPsip_nabla[0].data());
    
    ids["Nabla_GradPsipZ"]=0;
    nc_def_var( ncid, "Nabla_GradPsipZ", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Nabla_GradPsipZ"));
    nc_put_vara_double( ncid, ids.at("Nabla_GradPsipZ"),start, count, gradPsip_nabla[1].data());
    
    ids["Nabla_GradPsip_2"]=0;
    nc_def_var( ncid, "Nabla_GradPsip_2", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Nabla_GradPsip_2"));
    nc_put_vara_double( ncid, ids.at("Nabla_GradPsip_2"),start, count, gradPsip_2_nabla.data());
    
    ids["Nabla_LaplacePsip"]=0;
    nc_def_var( ncid, "Nabla_LaplacePsip", NC_DOUBLE, 2, &dim_ids[1], &ids.at("Nabla_LaplacePsip"));
    nc_put_vara_double( ncid, ids.at("Nabla_LaplacePsip"),start, count, deltaPsip_nabla.data());
    
    nc_close(ncid);
    return 0;
}
