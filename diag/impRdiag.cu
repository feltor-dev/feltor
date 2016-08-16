#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "dg/algorithm.h"
#include "dg/backend/xspacelib.cuh"
#include "file/nc_utilities.h"
#include "impurities/parameters.h"

// #include "dg/poisson.h"
// #include "dg/backend/interpolation.cuh"
// #include "dg/functors.h"
// #include "file/read_input.h"
// #include "probes.h"

double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}

struct Heaviside2d
{   Heaviside2d( double sigma):sigma2_(sigma*sigma), x_(0), y_(0){}
    void set_origin( double x0, double y0){ x_=x0, y_=y0;}
    double operator()(double x, double y)
    {   double r2 = (x-x_)*(x-x_)+(y-y_)*(y-y_);
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
{   if( argc != 3)
    {   std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.nc]\n";
        return -1;
    }
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
    const imp::Parameters p( js);
    p.display( std::cout);
    err = nc_close( ncid);
    ////////////////////////////////////////////////////////////////////////////
    //Grids
    dg::Grid2d<double > g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    dg::Grid1d<double > g1d( 0., p.lx, p.n_out, p.Nx_out, p.bc_x);
    dg::ArakawaX< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> arakawa( g2d); 
    //2d field
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[4] = {"electrons", "ions", "impurities", "potential"};
    int dataIDs[4];

    std::vector<dg::DVec> npe(3, dg::evaluate(dg::zero,g2d));
    std::vector<dg::DVec> ntilde(3, dg::evaluate(dg::zero,g2d));
    std::vector<dg::DVec> lnn(3, dg::evaluate(dg::zero,g2d));
    dg::DVec phi(dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> npe_h(3,dg::evaluate(dg::zero,g2d));
    dg::HVec phi_h(dg::evaluate(dg::zero,g2d));
    dg::DVec xvec = dg::evaluate( dg::coo1, g2d);
    dg::DVec yvec = dg::evaluate( dg::coo2, g2d);
    dg::HVec xcoo = dg::evaluate(dg::coo1,g1d);
    dg::DVec one = dg::evaluate( dg::one, g2d);
    dg::DVec w2d = dg::create::weights( g2d);
    dg::DVec helper(dg::evaluate(dg::zero,g2d));
    dg::DVec helper1d(dg::evaluate(dg::zero,g1d));
    dg::HVec transfer2d(dg::evaluate(dg::zero,g2d));
    dg::HVec transfer1d(dg::evaluate(dg::zero,g1d));
    dg::IDMatrix equi = dg::create::backscatter( g2d);

    double deltaT = p.dt*p.itstp;
    size_t count0d[1] = {1};
    size_t start0d[1] = {0};

    file::NC_Error_Handle err_out;
    int ncid_out, tvarID1d, dim_ids2d[3];
    err_out = nc_create( argv[2], NC_NETCDF4|NC_CLOBBER, &ncid_out);
    err_out = nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile", input.size(), input.data());
    err_out = file::define_limited_time( ncid_out, "time", p.maxout+1, &dim_ids2d[0], &tvarID1d);

    const size_t no_evar = 5;
    int evarID[no_evar];
    std::string evar[no_evar] = {"Se", "Si", "Sz", "Uphii", "Uphiz"};
    for( unsigned i=0; i<no_evar; i++)
        err_out = nc_def_var( ncid_out, evar[i].data(), NC_DOUBLE, 1, &dim_ids2d[0], &evarID[i]);
    const size_t nos = 3;
    int speciesID[nos];
    std::string species[nos] = {"electrons", "ions", "impurities"};
    const size_t no_povar = 14;
    int povarID[no_povar];
    std::string povar[no_povar] = {
        "posX" , "posY" , "velX" , "velY" , "accX" ,
        "accY","posXmax","posYmax","velXmax" , "velYmax",
        "compactness_ne", "velCOM", "maxamp", "mass"};
    for ( unsigned i=0; i<nos; i++)
    {   err_out = nc_def_grp( ncid_out, species[i].data(), &speciesID[i]);
        for ( unsigned j=0; j<no_povar; j++)
            err_out = nc_def_var( speciesID[i], povar[j].data(), NC_DOUBLE, 1, &dim_ids2d[0], &povarID[j]);
    }
    err_out = nc_close(ncid_out);

    // grid variables
    const double hx = g2d.hx()/(double)g2d.n();
    const double hy = g2d.hy()/(double)g2d.n();
    unsigned Nx = p.Nx_out*p.n_out; 
    //BLOB COMPACTNESS Heaviside function
    Heaviside2d heavi(2.0* p.sigma);
    double normalize = 1.;
    dg::DVec heavy;
    // species variables
    double mass_[nos]={};
    double posX[nos]={}, posY[nos]={};
    double posX_init[nos]={}, posY_init[nos]={};
    double posX_old[nos]={} ,posY_old[nos]={};
    double posX_max[nos]={}, posY_max[nos]={};
    double posX_max_old[nos]={}, posY_max_old[nos]={};
    double posX_max_hs[nos]={}, posY_max_hs[nos]={};
    double velX[nos]={}, velY[nos]={};
    double velX_old[nos]={}, velY_old[nos]={};
    double velX_max[nos]={}, velY_max[nos]={};
    double velCOM[nos]={};
    double accX[nos]={}, accY[nos]={};
    double compactness[nos] = {};
    unsigned position[nos] = {};
    double maxamp[nos];

    const dg::DVec binv( dg::evaluate( dg::LinearX( p.kappa, 1.), g2d));
    dg::DVec chi = dg::evaluate( dg::zero, g2d);
    dg::DVec gamma_n = dg::evaluate( dg::zero, g2d);
    dg::Helmholtz< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> gamma_s( g2d, 1.0, dg::centered);
    dg::Elliptic< dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol( g2d, dg::normed, dg::centered);
    dg::Invert< dg::DVec > invert_invgamma( chi, chi.size(), p.eps_gamma);
    ///////////////////////////Start timestepping///////////////////////////////
    double time = 0.;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);   
    err_out = nc_open( argv[2], NC_WRITE, &ncid_out);

    for( unsigned i=0; i<=p.maxout-1; i++)
    {   start2d[0] = i;
        start0d[0] = i;
        if (i>0)
            time += p.itstp*p.dt;

        err = nc_inq_varid( ncid, names[3].data(), &dataIDs[3]);
        err = nc_get_vara_double( ncid, dataIDs[3], start2d, count2d, phi_h.data());
        phi = phi_h;

        for (unsigned j=0; j<3; j++)
        {   err = nc_inq_varid( ncid, names[j].data(), &dataIDs[j]);
            err = nc_get_vara_double( ncid, dataIDs[j], start2d, count2d, npe_h[j].data());
            npe[j] = npe_h[j];
            dg::blas1::plus(npe[j], 1);
            dg::blas1::transform(npe[j], ntilde[j], dg::PLUS<double>(-1));

            if( j>1 && j<nos-1)
            {   gamma_s.alpha() = -0.5*p.tau[j]*p.mu[j];
                invert_invgamma( gamma_s, gamma_n, ntilde[j]);

                dg::blas1::axpby( -p.a[j]*p.mu[j], ntilde[j], 0., chi);
                dg::blas1::pointwiseDot( chi, binv, chi);
                dg::blas1::pointwiseDot( chi, binv, chi);
                pol.set_chi( chi);
                pol.symv( phi, ntilde[j]);
                dg::blas1::axpby( 1., gamma_n, 1., ntilde[j]);
            }

            err_out = nc_inq_grp_ncid( ncid_out, species[j].data(), &speciesID[j]);
            err_out = nc_inq_varids(speciesID[j], 0, povarID);

            //mass
            mass_[j] = dg::blas2::dot( one, w2d, ntilde[j] );
            err_out = nc_put_vara_double( speciesID[j], povarID[13], start0d, count0d, &mass_[j]);

            //position, velocity, acceleration
            if (i==0)
            {   posX_init[j] = dg::blas2::dot( xvec, w2d, ntilde[0])/mass_[j];
                posY_init[j] = dg::blas2::dot( yvec, w2d, ntilde[0])/mass_[j];
            }
            if (i>0)
            {   posX[j] = dg::blas2::dot( xvec, w2d, ntilde[0])/mass_[j]-posX_init[j];
                posY[j] = dg::blas2::dot( yvec, w2d, ntilde[0])/mass_[j]-posY_init[j];
            }
            if (i==0)
            {   velX_old[j] = -posX[j]/deltaT;
                velY_old[j] = -posY[j]/deltaT;
                posX_old[j] = posX[j];
                posY_old[j] = posY[j];
            }

            velX[j] = (posX[j] - posX_old[j])/deltaT;
            velY[j] = (posY[j] - posY_old[j])/deltaT;
            velCOM[j]=sqrt(velX[j]*velX[j]+velY[j]*velY[j]);
            accX[j] = (velX[j] - velX_old[j])/deltaT;
            accY[j] = (velY[j] - velY_old[j])/deltaT;
            if (i>0)
            {   posX_old[j] = posX[j]; posY_old[j] = posY[j];
                velX_old[j] = velX[j]; velY_old[j] = velY[j];
            }
            err_out = nc_put_vara_double( speciesID[j], povarID[0],  start0d, count0d, &posX[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[1],  start0d, count0d, &posY[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[2],  start0d, count0d, &velX[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[3],  start0d, count0d, &velY[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[4],  start0d, count0d, &accX[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[5],  start0d, count0d, &accY[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[11], start0d, count0d, &velCOM[j]);

            // maximal amplitude
            if ( p.amp > 0)
                maxamp[j] = *thrust::max_element( npe[j].begin(), npe[j].end());
            else
                maxamp[j] = *thrust::min_element( npe[j].begin(), npe[j].end());
            err_out = nc_put_vara_double( speciesID[j], povarID[12], start0d, count0d, &maxamp[j]);

            //get max position and value(x,y_max)
            dg::blas2::gemv( equi, npe[j], helper);
            position[j] = thrust::distance( helper.begin(), thrust::max_element( helper.begin(), helper.end()) );
            posX_max[j] = hx*(1./2. + (double)(position[j]%Nx))-posX_init[j];
            posY_max[j] = hy*(1./2. + (double)(position[j]/Nx))-posY_init[j];
            posX_max_hs[j] = hx*(1./2. + (double)(position[j]%Nx));
            posY_max_hs[j] = hy*(1./2. + (double)(position[j]/Nx));
            err_out = nc_put_vara_double( speciesID[j], povarID[6], start0d, count0d, &posX_max[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[7], start0d, count0d, &posY_max[j]);

            velX_max[j] = (posX_max[j] - posX_max_old[j])/deltaT;
            velY_max[j] = (posY_max[j] - posY_max_old[j])/deltaT;
            if (i>0)
                posX_max_old[j] = posX_max[j]; posY_max_old[j] = posY_max[j];
            err_out = nc_put_vara_double( speciesID[j], povarID[8], start0d, count0d, &velX_max[j]);
            err_out = nc_put_vara_double( speciesID[j], povarID[9], start0d, count0d, &velY_max[j]);

            if( i==0) std::cout << "COM: t = " << time << " amp :" << maxamp[j] \
                                << " X_init :" << posX_init[j] \
                                << " Y_init :" << posY_init[j] << "\n";
            std::cout << "COM: t = "<< time << " amp :" << maxamp[j] \
                      << " velX :" << velX[j] << " velY :" << velY[j] \
                      << " X :" << posX[j] << " Y :" << posY[j] << "\n";

            //compactness
            if (i==0)
            {   heavi.set_origin( posX_max_hs[j], posY_max_hs[j] );
                heavy = dg::evaluate( heavi, g2d);
                normalize = dg::blas2::dot( heavy, w2d, ntilde[j]);
            }
            heavi.set_origin( posX_max_hs[j], posY_max_hs[j]);
            heavy = dg::evaluate( heavi, g2d);
            compactness[j] =  dg::blas2::dot( heavy, w2d, ntilde[j])/normalize ;
            err_out = nc_put_vara_double( speciesID[j], povarID[10], start0d, count0d, &compactness[j]);

        //energy
            dg::blas1::transform( npe[j], lnn[j], dg::LN<double>());
        }

        arakawa.variation(phi, helper);
        double Se = dg::blas2::dot( npe[0], w2d, lnn[0]);
        double Si = p.a[1]*p.tau[1]*dg::blas2::dot( npe[1], w2d, lnn[1]);
        double Sz = p.a[2]*p.tau[2]*dg::blas2::dot( npe[2], w2d, lnn[2]);
        double Uphii = 0.5*p.a[1]*p.mu[1]*dg::blas2::dot( npe[1], w2d, helper);
        double Uphiz = 0.5*p.a[2]*p.mu[2]*dg::blas2::dot( npe[2], w2d, helper);

        err_out = nc_put_vara_double( ncid_out, evarID[0], start0d, count0d, &Se);
        err_out = nc_put_vara_double( ncid_out, evarID[1], start0d, count0d, &Si);
        err_out = nc_put_vara_double( ncid_out, evarID[2], start0d, count0d, &Sz);
        err_out = nc_put_vara_double( ncid_out, evarID[3], start0d, count0d, &Uphii);
        err_out = nc_put_vara_double( ncid_out, evarID[4], start0d, count0d, &Uphiz);

        //time
        err_out = nc_put_vara_double( ncid_out, tvarID1d, start0d, count0d, &time);
    }
    err_out = nc_close(ncid_out);
    err = nc_close(ncid);
    return 0;
}

