#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "dg/algorithm.h"
#include "dg/poisson.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"


#include "file/nc_utilities.h"
#include "toefl/parameters.h"
// #include "probes.h"

double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}

struct Heaviside2d
{
    Heaviside2d( double sigma):sigma2_(sigma*sigma), x_(0), y_(0){}
//     Heaviside2d( double sigma):sigma2_(sigma*sigma), x_(0), y_(0){}
    void set_origin( double x0, double y0){ x_=x0, y_=y0;}
    double operator()(double x, double y)
    {
        double r2 = (x-x_)*(x-x_)+(y-y_)*(y-y_);
        if( r2 >= sigma2_)
            return 0.;
        return 1.;
    }
  private:
    const double sigma2_;
    double x_,y_;
};

/*! Diagnostics program for the toefl code. 
 *
 * It reads in the produced netcdf file and outputs a new netcdf file with timeseries of
 * posX: COM x-position
 * posY: COM y-position
 * velX: COM x-velocity
 * velY: COM y-velocity
 * accX: COM x-acceleration
 * accY: COM y-acceleration
 * velCOM: absolute value of the COM velocity
 * posXmax: maximum amplitude x-position
 * posYmax: maximum amplitude y-position
 * velXmax: maximum amplitude x-velocity
 * velYmax: maximum amplitude y-velocity
 * maxamp: value of the maximum amplitude
 * compactness_ne: compactness of the density field
 * Ue: entropy electrons
 * Ui: entropy ions
 * Uphi: exb energy
 * mass: mass of the blob without background
 */
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
    
    Json::Reader reader;
    Json::Value js;
    reader.parse( input, js, false);
    const Parameters p(js);
    p.display(std::cout);
    err = nc_close( ncid);
    
    ///////////////////////////////////////////////////////////////////////////
    //Grids
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    dg::Grid1d g1d( 0., p.lx, p.n_out, p.Nx_out, p.bc_x);
    dg::ArakawaX< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> arakawa( g2d); 
    double time = 0.;
    //2d field
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[3] = {"electrons", "ions", "potential"}; 
    int dataIDs[3];
  
    std::vector<dg::DVec> npe(2, dg::evaluate(dg::zero,g2d));
    std::vector<dg::DVec> ntilde(2, dg::evaluate(dg::zero,g2d));
    std::vector<dg::DVec> lnn(2, dg::evaluate(dg::zero,g2d));
    dg::DVec phi(dg::evaluate(dg::zero,g2d));
    //dg::DVec vor(dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> npe_h(2,dg::evaluate(dg::zero,g2d));
    dg::HVec phi_h(dg::evaluate(dg::zero,g2d));
    //dg::HVec vor_h(dg::evaluate(dg::zero,g2d));
    dg::DVec xvec = dg::evaluate( dg::cooX2d, g2d);
    dg::DVec yvec = dg::evaluate( dg::cooY2d, g2d);
    dg::HVec xcoo = dg::evaluate(dg::cooX1d,g1d);
    dg::DVec one = dg::evaluate( dg::one, g2d);
    dg::DVec w2d = dg::create::weights( g2d);
    dg::DVec helper(dg::evaluate(dg::zero,g2d));
    dg::DVec helper2(dg::evaluate(dg::zero,g2d));
    dg::DVec helper3(dg::evaluate(dg::zero,g2d));
    dg::DVec helper4(dg::evaluate(dg::zero,g2d));

    dg::DVec helper1d(dg::evaluate(dg::zero,g1d));
    
    dg::HVec transfer2d(dg::evaluate(dg::zero,g2d));
    dg::HVec transfer1d(dg::evaluate(dg::zero,g1d));
    dg::IDMatrix equi = dg::create::backscatter( g2d);


    
    double mass_=0.;
    double posX=0.,posY=0.,posX_init=0.,posY_init=0.,posX_old=0.,posY_old=0.;
    double velX,velY,velX_old=0. , velY_old=0.;    
    double accX,accY=0.;
    double deltaT = p.dt*p.itstp;
    size_t count0d[1]  = {1};
    size_t start0d[1]  = {0};    
    //size_t count1d[2]  = {1, g2d.n()*g2d.Nx()};
    //size_t start1d[2]  = {0, 0};    
    //1d netcdf output file    

    file::NC_Error_Handle err_out;
    const size_t number_of_names = 17;
    int ncid_out;
    int namescomID[number_of_names],tvarID1d;
    int dim_ids2d[3];
    std::string namescom[number_of_names] = {
        "posX" , "posY" , "velX" , "velY" , "accX" , 
        "accY","posXmax","posYmax","velXmax" , "velYmax",
        "compactness_ne","velCOM","maxamp", "Ue", "Ui", 
        "Uphi", "mass"  };
    //std::string names1d[2] = {"ne_max", "x_"};
    
    err_out = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err_out = nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile", input.size(), input.data());
    //err_out = file::define_dimensions( ncid_out, dim_ids2d, &tvarID1d, g2d);
    err_out = file::define_limited_time( ncid_out, "time", p.maxout+1, &dim_ids2d[0], &tvarID1d);
    //err_out = file::define_time( ncid_out, "ptime", &timeID, &timevarID);
    for( unsigned i=0; i<number_of_names; i++){
        err_out = nc_def_var( ncid_out, namescom[i].data(),  NC_DOUBLE, 1, &dim_ids2d[0], &namescomID[i]);
    }   
    //for( unsigned i=0; i<2; i++){
    //    err_out = nc_def_var( ncid_out, names1d[i].data(),  NC_DOUBLE, 2, dim_ids2d, &names1dID[i]);
    //}   
    err_out = nc_close(ncid_out);   
    
    const double hx = g2d.hx()/(double)g2d.n();
    const double hy = g2d.hy()/(double)g2d.n();
    unsigned Nx = p.Nx_out*p.n_out; 
    //BLOB COMPACTNESS Heaviside function
    Heaviside2d heavi(2.0* p.sigma);
    double normalize = 1.;
    dg::DVec heavy;
    //open netcdf files
    unsigned position = 0;
    double posX_max = 0.0,posY_max = 0.0,posX_max_old = 0.0,posY_max_old = 0.0,velX_max=0.0, velY_max=0.0,posX_max_hs=0.0,posY_max_hs=0.0,velCOM=0.0;
    double compactness_ne=0.0;
    //-----------------Start timestepping
    err = nc_open( argv[1], NC_NOWRITE, &ncid);   
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);
    for( unsigned i=0; i<=p.maxout; i++)
    {
        start2d[0] = i;
        //start1d[0] = i;
        start0d[0] = i;


        for (unsigned j=0;j<2;j++)
        {
            err = nc_inq_varid(ncid, names[j].data(), &dataIDs[j]);
            err = nc_get_vara_double( ncid, dataIDs[j], start2d, count2d, npe_h[j].data());
            npe[j] = npe_h[j];
            dg::blas1::plus(npe[j], 1);
        }
        err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
        err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d, phi_h.data());
        phi = phi_h;
        //err = nc_inq_varid(ncid, names[5].data(), &dataIDs[5]);
        //err = nc_get_vara_double( ncid, dataIDs[5], start2d, count2d, vor_h.data());
        //vor = vor_h;
        dg::blas1::transform(npe[0], ntilde[0], dg::PLUS<double>(-1));
        dg::blas1::transform(npe[1], ntilde[1], dg::PLUS<double>(-1));
        ///////////////////////////Compute mass
        mass_ = dg::blas2::dot( one, w2d, ntilde[0] ); 
        if (i==0){
            posX_init = dg::blas2::dot( xvec, w2d, ntilde[0])/mass_;
            posY_init = dg::blas2::dot( yvec, w2d, ntilde[0])/mass_;    
        }
        if (i>0){
            time += p.itstp*p.dt;

            posX = dg::blas2::dot( xvec, w2d, ntilde[0])/mass_-posX_init ;
            posY = dg::blas2::dot( yvec, w2d, ntilde[0])/mass_-posY_init;
        }
        if (i==0){
            velX_old = -posX/deltaT;
            velY_old = -posY/deltaT; 
            posX_old = posX;
            posY_old = posY;
        }
        
        velX = (posX - posX_old)/deltaT;
        velY = (posY - posY_old)/deltaT;
        velCOM=sqrt(velX*velX+velY*velY);
        accX = (velX - velX_old)/deltaT;
        accY = (velY - velY_old)/deltaT;
        if (i>0){
        posX_old = posX; posY_old = posY;
        velX_old = velX; velY_old = velY;   
        }

        
       err_out = nc_put_vara_double( ncid_out, namescomID[16], start0d, count0d, &mass_);
       err_out = nc_put_vara_double( ncid_out, namescomID[0], start0d, count0d, &posX);
       err_out = nc_put_vara_double( ncid_out, namescomID[1], start0d, count0d, &posY);
       err_out = nc_put_vara_double( ncid_out, namescomID[2], start0d, count0d, &velX);
       err_out = nc_put_vara_double( ncid_out, namescomID[3], start0d, count0d, &velY);
       err_out = nc_put_vara_double( ncid_out, namescomID[4], start0d, count0d, &accX);
       err_out = nc_put_vara_double( ncid_out, namescomID[5], start0d, count0d, &accY);
       err_out = nc_put_vara_double( ncid_out, namescomID[11], start0d, count0d, &velCOM);
       //err_out = nc_put_vara_double( ncid_out, timevarID, start0d, count0d, &time);
       err_out = nc_put_vara_double( ncid_out, tvarID1d, start0d, count0d, &time);         
       double maxamp;
       if ( p.amp > 0)
           maxamp = *thrust::max_element( npe[0].begin(), npe[0].end());
       else
           maxamp = *thrust::min_element( npe[0].begin(), npe[0].end());
       err_out = nc_put_vara_double( ncid_out, namescomID[12], start0d, count0d, &maxamp);
       
       //get max position and value(x,y_max) of electron density
        dg::blas2::gemv( equi, npe[0], helper);
        position = thrust::distance( helper.begin(), thrust::max_element( helper.begin(), helper.end()) );
        posX_max = hx*(1./2. + (double)(position%Nx))-posX_init;
        posY_max = hy*(1./2. + (double)(position/Nx))-posY_init;
        posX_max_hs = hx*(1./2. + (double)(position%Nx));
        posY_max_hs = hy*(1./2. + (double)(position/Nx));
//         std::cout << "posXmax "<<posX_max<<" posYmax "<<posY_max << std::endl;
        err_out = nc_put_vara_double( ncid_out, namescomID[6], start0d, count0d, &posX_max);
        err_out = nc_put_vara_double( ncid_out, namescomID[7], start0d, count0d, &posY_max);      
        velX_max = (posX_max - posX_max_old)/deltaT;
        velY_max = (posY_max - posY_max_old)/deltaT;
        if( i==0) std::cout << "COM: t = "<< time << " amp :" << maxamp << " X_init :" << posX_init << " Y_init :" << posY_init << "\n";
        std::cout << "COM: t = "<< time << " amp :" << maxamp << " velX :" << velX << " velY :" << velY << " X :" << posX << " Y :" << posY << "\n";
        //std::cout << "MAX: time = "<< time << " mass :" << mass_ << " velX :" << velX_max << " velY :" << velY_max << " posX :" << posX_max << " posY :" << posY_max << std::endl;
        if (i>0){
            posX_max_old = posX_max; posY_max_old = posY_max;
        }  
        err_out = nc_put_vara_double( ncid_out, namescomID[8], start0d, count0d, &velX_max);
        err_out = nc_put_vara_double( ncid_out, namescomID[9], start0d, count0d, &velY_max);      
//         std::cout << "maxval "<<*thrust::max_element( helper.begin(), helper.end())<< std::endl;        
        //Compute interpolation matrix for 1d field    
        //dg::HVec y0coone(dg::evaluate(dg::CONSTANT(posY_max_hs ),g1d));
        //dg::IDMatrix interpne(dg::create::interpolation(xcoo,y0coone, g2d)) ;
        //
        //dg::blas2::gemv(interpne,npe[0],helper1d); 
        //dg::blas1::transfer( helper1d, transfer1d);
        //err_out = nc_put_vara_double( ncid_out, names1dID[0], start1d, count1d, transfer1d.data());    
        
        
        ///////////////////BLOB COMPACTNESS/////////////////
        if (i==0) {
            heavi.set_origin( posX_max_hs, posY_max_hs );
            heavy = dg::evaluate( heavi, g2d);
            normalize = dg::blas2::dot( heavy, w2d, ntilde[0]);
        }
        heavi.set_origin( posX_max_hs, posY_max_hs);
        heavy = dg::evaluate( heavi, g2d);
//         std::cout <<std::scientific<< dg::blas2::dot( heavy, w2d, npe[0])/normalize << std::endl;
        compactness_ne =  dg::blas2::dot( heavy, w2d, ntilde[0])/normalize ;
        err_out = nc_put_vara_double( ncid_out, namescomID[10], start0d, count0d, &compactness_ne);            
        /////////////////BLOB energetics/////////////////
        double Ue, Ui, Uphi;
        for( unsigned j=0; j<2; j++)
            dg::blas1::transform( npe[j], lnn[j], dg::LN<double>()); 
        arakawa.variation(phi, helper); 
        if(p.equations == "global" || p.equations == "ralf_global")
        {
            Ue = dg::blas2::dot( lnn[0], w2d, npe[0]);
            Ui = p.tau*dg::blas2::dot( lnn[1], w2d, npe[1]);
            Uphi = 0.5*dg::blas2::dot( npe[1], w2d, helper); 
        }
        else
        {
            Ue = 0.5*dg::blas2::dot( ntilde[0], w2d, ntilde[0]);
            Ui = 0.5*p.tau*dg::blas2::dot( ntilde[1], w2d, ntilde[1]);
            Uphi = 0.5*dg::blas2::dot( one, w2d, helper); 
        }
        err_out = nc_put_vara_double( ncid_out, namescomID[13], start0d, count0d, &Ue);            
        err_out = nc_put_vara_double( ncid_out, namescomID[14], start0d, count0d, &Ui);            
        err_out = nc_put_vara_double( ncid_out, namescomID[15], start0d, count0d, &Uphi); 

    }
    err_out = nc_close(ncid_out);
    err = nc_close(ncid);
    
    return 0;
}

