#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "parameters.h"

int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.nc]\n";
        return -1;
    }
//     std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    ///////////////////read in and show inputfile//////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    std::cout << "input "<<input<<std::endl;
    auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
    const eule::Parameters p(js);
    p.display(std::cout);
    
    ///////////////////////////////////////////////////////////////////////////
    //Grids
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    double time = 0.;
    //2d field
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[6] = {"electrons", "ions", "Telectrons","Tions", "potential","vor"}; 
    int dataIDs[6];
    
    std::vector<dg::HVec> npe(2,dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> tpe(2,dg::evaluate(dg::zero,g2d));
    dg::HVec phi(dg::evaluate(dg::zero,g2d));
    dg::HVec vor(dg::evaluate(dg::zero,g2d));
    dg::HVec xvec = dg::evaluate( dg::cooX2d, g2d);
    dg::HVec yvec = dg::evaluate( dg::cooY2d, g2d);
    dg::HVec one = dg::evaluate( dg::one, g2d);
    dg::HVec w2d = dg::create::weights( g2d);
    
    double mass_=0.;
    double posX,posY,posX_init=0,posY_init=0,posX_old=0,posY_old=0.;
    double velX,velY,velX_old=0 , velY_old=0.;    
    double accX,accY=0.;
    double deltaT = p.dt*p.itstp;
    dg::Grid1d g1d( 0., p.lx,p.n_out, p.Nx_out, p.bc_x);
    size_t count1d[2]  = {1, g2d.n()*g2d.Nx()};
    size_t start1d[2]  = {0, 0};    
    //1d netcdf output file    

    dg::file::NC_Error_Handle err1d;
    int ncid1d,dim_ids1d[2], tvarID1d,namescomID[6],timeID, timevarID;
    err1d = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid1d);
    err1d = nc_put_att_text( ncid1d, NC_GLOBAL, "inputfile", input.size(), input.data());
    err1d = dg::file::define_dimensions( ncid1d, dim_ids1d, &tvarID1d, g1d);
    err1d = nc_close(ncid1d); 
    err1d = nc_open( argv[2], NC_WRITE, &ncid1d);
    err1d = nc_redef(ncid1d);

    err1d = dg::file::define_time( ncid1d, "ptime", &timeID, &timevarID);
    std::string namescom[6] = {"posX" , "posY" , "velX" , "velY" , "accX" , "accY" };

    for( unsigned i=0; i<6; i++){
        std::cout << i << std::endl;
        err1d = nc_def_var( ncid1d, namescom[i].data(),  NC_DOUBLE, 1, &timeID, &namescomID[i]);
    }   

    err1d = nc_enddef(ncid1d);   

//     err1d = nc_close(ncid1d); 

    
    
    //////////////////////////////open nc file//////////////////////////////////
    err1d = nc_open( argv[2], NC_WRITE, &ncid1d);

    for( unsigned i=0; i<p.maxout; i++)//timestepping
    {
        start2d[0] = i;
        start1d[0] = i;

        time += p.itstp*p.dt;

        std::cout << "time = "<< time <<  std::endl;
        for (unsigned j=0;j<2;j++)
        {
            err = nc_inq_varid(ncid, names[j].data(), &dataIDs[j]);
            err = nc_get_vara_double( ncid, dataIDs[j], start2d, count2d, npe[j].data());
            dg::blas1::transform(npe[j], npe[j], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
        }
        for (unsigned j=0;j<2;j++)
        {
            err = nc_inq_varid(ncid, names[j+2].data(), &dataIDs[j+2]);
            err = nc_get_vara_double( ncid, dataIDs[j+2], start2d, count2d, tpe[j].data());
            dg::blas1::transform(tpe[j], tpe[j], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
        }
        err = nc_inq_varid(ncid, names[4].data(), &dataIDs[4]);
        err = nc_get_vara_double( ncid, dataIDs[4], start2d, count2d, phi.data());
        err = nc_inq_varid(ncid, names[5].data(), &dataIDs[5]);
        err = nc_get_vara_double( ncid, dataIDs[5], start2d, count2d, vor.data());
        mass_ = dg::blas2::dot( one, w2d, npe[0] ); 
        if (i==0){
            posX_init = dg::blas2::dot( xvec, w2d, npe[0])/mass_;
            posY_init = dg::blas2::dot( yvec, w2d, npe[0])/mass_;    
        }
        posX = dg::blas2::dot( xvec, w2d, npe[0])/mass_ - posX_init;
        posY = dg::blas2::dot( yvec, w2d, npe[0])/mass_ - posY_init;
        if (i==0){
            velX_old = -posX/deltaT;
            velY_old = -posY/deltaT; 
            posX_old = posX;
            posY_old = posY;
        }
        velX = (posX - posX_old)/deltaT;
        velY = (posY - posY_old)/deltaT;
        accX = (velX - velX_old)/deltaT;
        accY = (velY - velY_old)/deltaT;
        posX_old = posX; posY_old = posY;
        velX_old = velX; velY_old = velY;   

        std::cout << "mass :" << mass_ << " velX :" << velX << " velY :" << velY << " posX :" << posX << " posY :" << posY << std::endl;
        
       err1d = nc_put_vara_double( ncid1d, timevarID, start1d, count1d, &time);
       err1d = nc_put_vara_double( ncid1d, namescomID[0], start1d, count1d, &posX);
       err1d = nc_put_vara_double( ncid1d, namescomID[1], start1d, count1d, &posY);
       err1d = nc_put_vara_double( ncid1d, namescomID[2], start1d, count1d, &velX);
       err1d = nc_put_vara_double( ncid1d, namescomID[3], start1d, count1d, &velY);
       err1d = nc_put_vara_double( ncid1d, namescomID[4], start1d, count1d, &accX);
       err1d = nc_put_vara_double( ncid1d, namescomID[5], start1d, count1d, &accY);
               err1d = nc_put_vara_double( ncid1d, tvarID1d, start1d, count1d, &time);                    

    }
    err1d = nc_close(ncid1d);
    err = nc_close(ncid);
    
    return 0;
}

