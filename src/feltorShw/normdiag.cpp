#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <thrust/extrema.h>

#include "dg/algorithm.h"
#include "dg/poisson.h"

#include "dg/functors.h"

#include "dg/file/file.h"
#include "parameters.h"
int main( int argc, char* argv[])
{
    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc]\n";
        return -1;
    }

    //////////////////////////////open nc file//////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    ///////////////////read in and show inputfile //////////////////
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);    
    err = nc_close(ncid); 

    auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
    const eule::Parameters p(js);
    
    //////////////////////////////Grids//////////////////////////////////////
    //input grid
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    
    //2d field netcdf vars of input.nc
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    int dataIDs[4];
    
    //dg stuff
    dg::HVec tempH(dg::evaluate(dg::one,g2d));
    dg::DVec one(dg::evaluate(dg::one,g2d));
    dg::DVec phi(dg::evaluate(dg::zero,g2d));
    dg::DVec ne(dg::evaluate(dg::zero,g2d));
    dg::DVec logne(dg::evaluate(dg::zero,g2d));
    dg::DVec lognednG(dg::evaluate(dg::zero,g2d));
    dg::DVec Ni(dg::evaluate(dg::zero,g2d));
    dg::DVec dNi(dg::evaluate(dg::zero,g2d));
    dg::DVec dne(dg::evaluate(dg::zero,g2d));
    dg::DVec uE2(dg::evaluate(dg::zero,g2d));
    dg::ExpProfX prof(p.nprofileamp, p.bgprofamp,p.invkappa); 
    dg::DVec nG(dg::evaluate(prof,g2d));
    dg::DVec w2d = dg::create::weights( g2d);
    dg::Poisson<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> poisson(g2d,  p.bc_x, p.bc_y,  p.bc_x_phi, p.bc_y);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol(g2d, p.bc_x_phi, p.bc_y);
    //open netcdf files
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    //set min and max timesteps
    double time = 0.;
    unsigned imin,imax;    
    imin=0; //set min time    
    //get max time
    err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]); 
    size_t steps;
    err = nc_inq_dimlen(ncid, dataIDs[0], &steps);
    steps-=1;
    imax = steps/p.itstp;
    double deltaT = p.dt*p.itstp;     //define timestep

    

    double phisupnorm=0., phisupnorm0=0., phinorm=0., phinorm0=0., dnesupnorm=0., dnesupnorm0=0., dnenorm=0., dnenorm0=0.,  Enorm=0., Enorm0=0., Eqnorm=0., Eqnorm0=0., Edfnorm=0., Edfnorm0=0., uE2norm=0., NiuE2norm=0., nlnnnorm=0., NiuE2normq=0., nlnnnormq=0.,  unorm=0., unorm0=0., usupnorm=0., usupnorm0=0.;

    for( unsigned i=imin; i<=imax; i++)//timestepping
    {
        
        start2d[0] = i;
        //get input.nc data
        err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
        err = nc_inq_varid(ncid, names[1].data(), &dataIDs[1]);
        err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
        err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, tempH.data());
        if (p.modelmode==0 || p.modelmode==1)
        {
            ne=tempH;
            dg::blas1::axpby(1.0,ne,+1.0,one,ne);
            dg::blas1::transform(ne, logne, dg::LN<double>());

        }
        if (p.modelmode==2)
        {
            dne=tempH;
            dg::blas1::axpby(1.0,one,+1.0,dne,ne);
            dg::blas1::pointwiseDivide(ne,nG,ne);
            dg::blas1::transform(ne, logne, dg::LN<double>());
            
        }
        if (p.modelmode==3)
        {
            dne=tempH;
            dg::blas1::transform(dne, dne, dg::EXP<double>());
            dg::blas1::transform(dne, dne,  dg::PLUS<>(-1.0)); 
            dg::blas1::axpby(1.0,one,+1.0,dne,ne);
            dg::blas1::pointwiseDivide(ne,nG,ne);
            dg::blas1::transform(ne, logne, dg::LN<double>());
            
        }
        err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, tempH.data());
        if (p.modelmode==0 || p.modelmode==1)
        {
            Ni=tempH;
            dg::blas1::axpby(1.0,Ni,+1.0,one,Ni);        
            dg::blas1::pointwiseDivide(Ni,nG,dNi);
            dg::blas1::axpby(1.0,dNi,-1.0,one,dNi);
        } 
        if (p.modelmode==2)
        {
            dNi=tempH;
            dg::blas1::axpby(1.0,one,+1.0,dNi,Ni);
            dg::blas1::pointwiseDivide(Ni,nG,Ni);
        }
        if (p.modelmode==3)
        {
            dNi=tempH;
            dg::blas1::transform(dNi, dNi, dg::EXP<double>());
            dg::blas1::transform(dNi, dNi,  dg::PLUS<>(-1.0)); 
            dg::blas1::axpby(1.0,one,+1.0,dNi,Ni);
            dg::blas1::pointwiseDivide(Ni,nG,Ni);
        }
        err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d, tempH.data());
        phi=tempH;

        //get max phi value

        phisupnorm=*thrust::max_element(phi.begin(),phi.end());
        phinorm = sqrt(dg::blas2::dot(phi,w2d,phi));
        if (p.modelmode==2 || p.modelmode==3)
        {
            dnesupnorm =*thrust::max_element(dne.begin(),dne.end()); 
            dnenorm = sqrt(dg::blas2::dot(dne,w2d,dne));
        }

        dg::blas1::pointwiseDivide(ne,nG,dne);
        dg::blas1::transform(dne, lognednG, dg::LN<double>());
        dg::blas1::pointwiseDot(lognednG,dne,lognednG);
        dg::blas1::axpby(1.0,dne,-1.0,one,dne);
        dg::blas1::axpby(1.0,lognednG,-1.0,dne,lognednG);
        nlnnnormq = dg::blas2::dot(one,w2d,lognednG);

        if (p.modelmode==0 || p.modelmode==1)
        {
            dnesupnorm =*thrust::max_element(dne.begin(),dne.end());
            dnenorm = sqrt(dg::blas2::dot(dne,w2d,dne));
        }
        

        pol.variation(phi,uE2);
        uE2norm= 0.5*dg::blas2::dot( one, w2d,uE2);   // 0.5   u_E^2    
        nlnnnorm = dg::blas2::dot(ne,w2d,logne);
        NiuE2norm = 0.5*dg::blas2::dot(Ni, w2d,uE2);
        dg::blas1::pointwiseDivide(Ni,nG,dNi);
        NiuE2normq = 0.5*dg::blas2::dot(Ni, w2d,uE2);
        
        
        Edfnorm = uE2norm + 0.5*dnenorm*dnenorm;
        Enorm = NiuE2norm + nlnnnorm ;
        Eqnorm = NiuE2normq + nlnnnormq;

        unorm = sqrt(dnenorm*dnenorm + phinorm*phinorm);
        usupnorm = sqrt(dnesupnorm*dnesupnorm + phisupnorm*phisupnorm);
        //normalize
        if (i==0) {
            phisupnorm0=1.+phisupnorm; //is zero at t=0
            phinorm0=1.+phinorm; //is zero at t=0
            dnesupnorm0=dnesupnorm;
            dnenorm0=dnenorm;
            Edfnorm0=Edfnorm;
            Enorm0=Enorm;
            Eqnorm0=Eqnorm;
            unorm0=unorm;
            usupnorm0=usupnorm;
        }
        //write norm data
        std::cout << time << " " <<  (1.+phisupnorm)/phisupnorm0<< " " 
                                 <<  dnesupnorm/dnesupnorm0<< " "
                                 <<  (1.+phinorm)/phinorm0<< " " 
                                 <<  dnenorm/dnenorm0<< " "
                                 <<  Edfnorm/Edfnorm0<<" "
                                 <<  Enorm/Enorm0<<" "
                                 <<  Eqnorm/Eqnorm0<<" " 
                                 <<  unorm/unorm0<<" "
                                 <<  usupnorm/usupnorm0<<"\n";

        time += deltaT;
    }


    err = nc_close(ncid);
    return 0;
}


