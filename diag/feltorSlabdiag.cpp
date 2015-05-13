#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/algorithm.h"
#include "dg/poisson.h"

#include "dg/backend/interpolation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/average.cuh"
#include "dg/functors.h"

#include "file/read_input.h"
#include "file/nc_utilities.h"
#include "feltorSlab/parameters.h"

int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.nc]\n";
        return -1;
    }
//     std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    //////////////////////////////open nc file//////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    ///////////////////read in and show inputfile und geomfile//////////////////
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    
    std::cout << "input "<<input<<std::endl;
    
    const eule::Parameters p(file::read_input( input));
    p.display();
    
    ///////////////////////////////////////////////////////////////////////////
    //Grids
    dg::Grid2d<double > g2d( 0., p.lx, 0.,p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::Grid1d<double > g1d( 0., p.lx,p.n, p.Nx, p.bc_x);
    double time = 0.;

    std::vector<dg::HVec> npe(2,dg::evaluate(dg::zero,g2d));
    dg::HVec phi(dg::evaluate(dg::zero,g2d));
    dg::HVec vor(dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> logn(2,dg::evaluate(dg::zero,g2d));
    dg::HVec temp(dg::evaluate(dg::zero,g2d));
    dg::HVec temp1d(dg::evaluate(dg::zero,g1d));
    dg::HVec xcoo(dg::evaluate(dg::coo1,g1d));
//     dg::HVec y0coo(dg::evaluate(1,0.0));
    dg::HVec y0coo(dg::evaluate(dg::CONSTANT(0.0),g1d));
    dg::PoloidalAverage<dg::HVec,dg::HVec > polavg(g2d);
    dg::HMatrix interp(dg::create::interpolation(xcoo,y0coo,g2d));
    
    //2d field
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    int dataIDs[4]; 
    //1d profiles
    file::NC_Error_Handle err1d;
    int ncid1d,dim_ids1d[2],dataIDs1d[7], tvarID1d;
    std::string names1d[7] =  {"neavg", "Niavg",  "ln(ne)avg","ln(Ni)avg","potentialavg","voravg","x_"}; 
    size_t count1d[2]  = {1, g2d.n()*g2d.Nx()};
    size_t start1d[2]  = {0, 0};    
    err1d = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid1d);
    err1d= nc_put_att_text( ncid1d, NC_GLOBAL, "inputfile", input.size(), input.data());
    err1d= file::define_dimensions( ncid1d, dim_ids1d, &tvarID1d, g1d);

    for( unsigned i=0; i<7; i++){
        std::cout << i << std::endl;
        err1d = nc_def_var( ncid1d, names1d[i].data(), NC_DOUBLE, 2, dim_ids1d, &dataIDs1d[i]);
    }   
    err1d = nc_close(ncid1d); 
    //2d field netcdf vars read

    
    unsigned imin,imax;
    std::cout << "tmin = 0 tmax =" << p.maxout*p.itstp << std::endl;
    std::cout << "enter new imin(>0) and imax(<maxout):" << std::endl;
    std::cin >> imin >> imax;
    time = imin*p.itstp;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    err1d = nc_open( argv[2], NC_WRITE, &ncid1d);

    for( unsigned i=imin; i<imax; i++)//timestepping
    {
            start2d[0] = i;
            start1d[0] = i;
            time += p.itstp*p.dt;

            std::cout << "time = "<< time <<  std::endl;

            err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
            err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, npe[0].data());
            err = nc_inq_varid(ncid, names[1].data(), &dataIDs[1]);
            err = nc_get_vara_double( ncid, dataIDs[1], start2d, count2d, npe[1].data());
            err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
            err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d, phi.data());
            err = nc_inq_varid(ncid, names[3].data(), &dataIDs[3]);
            err = nc_get_vara_double( ncid, dataIDs[3], start2d, count2d, vor.data());
            dg::blas1::transform(npe[0], npe[0], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
            dg::blas1::transform(npe[1], npe[1], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
            dg::blas1::transform( npe[0], logn[0], dg::LN<double>());
            dg::blas1::transform( npe[1], logn[1], dg::LN<double>());

            polavg(npe[0],temp);
            dg::blas2::gemv(interp,temp,temp1d); 
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[0],   start1d, count1d, temp1d.data()); 
            polavg(npe[1],temp);
            dg::blas2::gemv(interp,temp,temp1d); 
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[1],   start1d, count1d, temp1d.data()); 
            polavg(logn[0],temp);
            dg::blas2::gemv(interp,temp,temp1d); 
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[2],   start1d, count1d, temp1d.data()); 
            polavg(logn[1],temp);
            dg::blas2::gemv(interp,temp,temp1d); 
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[3],   start1d, count1d, temp1d.data()); 
            polavg(phi,temp);
            dg::blas2::gemv(interp,temp,temp1d); 
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[4],   start1d, count1d, temp1d.data()); 
            polavg(vor,temp);
            dg::blas2::gemv(interp,temp,temp1d); 
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[5],   start1d, count1d, temp1d.data()); 

            err1d = nc_put_vara_double( ncid1d, dataIDs1d[6],   start1d, count1d, xcoo.data()); 
            err1d = nc_put_vara_double( ncid1d, tvarID1d, start1d, count1d, &time);            
        
    }
    err = nc_close(ncid);
    
    err1d = nc_close(ncid1d);
    return 0;
}

