#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include "spectral/drt_dft.h"
#include "spectral/drt_drt.h"

#include "dg/algorithm.h"
#include "dg/poisson.h"

#include "dg/backend/interpolation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/average.cuh"
#include "dg/functors.h"

#include "file/nc_utilities.h"
#include "feltorShw/parameters.h"
int main( int argc, char* argv[])
{
    if( argc != 4)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [outputkxky.nc] [outputk.nc]\n";
        return -1;
    }
//     std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    //////////////////////////////open nc file//////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    ///////////////////read in and show inputfile //////////////////
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);    
    err = nc_close(ncid); 

    std::cout << "input "<<input<<std::endl;    
    Json::Reader reader;
    Json::Value js;
    reader.parse( input, js, false);
    const eule::Parameters p(js);
    p.display(std::cout);
    
    ///////////////////////////////////////////////////////////////////////////
    //Grids
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    const size_t rows =  g2d.n()*g2d.Ny();
    const size_t cols = g2d.n()*g2d.Nx();
    
    const double kxmin = 1./p.lx;
    const double kxmax = (cols/2+1)/p.lx;
    const unsigned Nkx = cols/2+1;
    const double kymin = 1./p.ly;
    const double kymax = (rows/2+1)/p.ly;
    const unsigned Nky = rows/2+1;

    
    dg::Grid2d g2d_f( kxmin, kxmax, kymin, kymax, 1., Nkx, Nky, p.bc_x, p.bc_y);
    dg::Grid1d g1d_f( kxmin,kxmax,1., Nkx,  p.bc_y);

    dg::Poisson<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> poisson(g2d,  g2d.bcx(), g2d.bcy(),  g2d.bcx(), g2d.bcy());

    //2d field netcdf vars read
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};

    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    int dataIDs[4]; 
    //2d file
    file::NC_Error_Handle err2d_f;
    int ncid2d_f,dim_ids2d_f[3],dataIDs2d_f[5], tvarID2d_f;
    std::string names2d_f[6] = {"S(Ue)","S(Ui)","S(UE)","S(ne)","S(phi)","gamma(phi)"}; //may  goto ln(n/<n>)
    size_t count2d_f[3]  = {1, g2d_f.Ny(), g2d_f.Nx()};
    size_t start2d_f[3]  = {0, 0, 0};    
    err2d_f = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid2d_f);
    err2d_f = nc_put_att_text( ncid2d_f, NC_GLOBAL, "inputfile", input.size(), input.data());
    err2d_f = file::define_dimensions( ncid2d_f, dim_ids2d_f, &tvarID2d_f, g2d_f);
    for( unsigned i=0; i<6; i++){
        err2d_f = nc_def_var( ncid2d_f, names2d_f[i].data(), NC_DOUBLE, 3, dim_ids2d_f, &dataIDs2d_f[i]);
    }   
    err2d_f = nc_close(ncid2d_f); 
    //1d file
    dg::HVec kn= dg::evaluate(dg::cooX1d,g1d_f);
    file::NC_Error_Handle err1d_f;
    int ncid1d_f,dim_ids1d_f[2],dataIDs1d_f[6], tvarID1d_f;
    std::string names1d_f[7] = {"Sk(Ue)","Sk(Ui)","Sk(UE)","Sk(ne)","Sk(phi)","gamma(phi)","k",}; //may  goto ln(n/<n>)
    size_t count1d_f[2]  = {1, g2d_f.Nx()};
    size_t start1d_f[2]  = {0, 0};    
    err1d_f = nc_create(argv[3],NC_NETCDF4|NC_CLOBBER, &ncid1d_f);
    err1d_f = nc_put_att_text( ncid1d_f, NC_GLOBAL, "inputfile", input.size(), input.data());
    err1d_f = file::define_dimensions( ncid1d_f, dim_ids1d_f, &tvarID1d_f, g1d_f);
    for( unsigned i=0; i<7; i++){
        err1d_f = nc_def_var( ncid1d_f, names1d_f[i].data(), NC_DOUBLE, 2, dim_ids1d_f, &dataIDs1d_f[i]);
    }   
    err1d_f = nc_close(ncid1d_f); 
    
    std::vector<dg::HVec> npe(2,dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> ntilde(2,dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> logn(2,dg::evaluate(dg::zero,g2d));
    dg::HVec phi(dg::evaluate(dg::zero,g2d));
    dg::HVec uE(dg::evaluate(dg::zero,g2d));
    dg::HVec one(dg::evaluate(dg::one,g2d));
    dg::HVec nprof(dg::evaluate(dg::ExpProfX(p.nprofileamp, p.bgprofamp,p.invkappa),g2d));
    std::vector<dg::HVec> energies(5,phi); //Se,Si,SE
    std::vector<dg::HVec> energiesequi(5,phi); 

    double time = 0.;
    dg::IHMatrix equi = dg::create::backscatter( g2d);
 
    //fftw setup

    spectral::Matrix<double, spectral::TL_DRT_DFT> temp(  rows, cols);
    spectral::Matrix<std::complex<double> > comspec( cols,rows/2 +1 );
    spectral::Matrix<double > squspec( g2d_f.Nx(), g2d_f.Ny());
    spectral::Matrix<double > gammasquspec( g2d_f.Nx(), g2d_f.Ny());
    
    std::vector<double > gammashellspec( g2d_f.Nx());    
    std::vector<double > shellspec( g2d_f.Nx());
    std::vector<unsigned > counter( g2d_f.Nx());
    std::vector<double > kxspec( g2d_f.Nx());
    std::vector<double > kyspec( g2d_f.Ny());
    std::vector<double > tempx( g2d_f.Nx());
    std::vector<double > tempy( g2d_f.Ny());
    std::vector<double > compspecx( g2d_f.Nx());
    std::vector<double > compspecy( g2d_f.Ny());
    std::vector<double > abscompspecx( g2d_f.Nx());
    std::vector<double > abscompspecy( g2d_f.Ny());

    
    //FFTW_RODFT11 computes an RODFT11 transform, i.e. a DST-IV. (Logical N=2*n, inverse is FFTW_RODFT11.)  -> DIR_NEU
    // FFTW_RODFT10 computes an RODFT10 transform, i.e. a DST-II. (Logical N=2*n, inverse is FFTW_RODFT01.) -> DIR_DIR
    //FFTW_RODFT00 computes an RODFT00 transform, i.e. a DST-I. (Logical N=2*(n+1), inverse is FFTW_RODFT00.) 
    fftw_r2r_kind kind = FFTW_RODFT00; //DST & DST IV
    spectral::DRT_DFT drt_dft( rows, cols, kind);
//     hindfty = fftw_plan_dft_1d(g2d_f.Ny(), tempy, compspecy, FFTW_FORWARD,FFTW_ESTIMATE); //DST
//     hindftx = fftw_plan_r2r_1d(nx-2,       tempx, compspecx, FFTW_RODFT11,FFTW_ESTIMATE); //DST IV

    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    err2d_f = nc_open( argv[2], NC_WRITE, &ncid2d_f);
    err1d_f = nc_open( argv[3], NC_WRITE, &ncid1d_f);
    unsigned imin,imax;
    
    //set min time
    imin=0;
    //get max time
    err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
    size_t steps;
    err = nc_inq_dimlen(ncid, dataIDs[0], &steps);
    steps-=1;
    imax = steps/p.itstp;
        
    //define timestep
    double deltaT = p.dt*p.itstp;

    //initialize gammas
    for( size_t m = 0; m < g2d_f.Nx(); m++) {
        for( size_t n = 0; n < g2d_f.Ny(); n++) {
            gammasquspec(m,n) =0.;
        }
    }
    for (size_t mn=0;mn<(unsigned)g2d_f.Nx();mn++) {
            gammashellspec[mn]=0.;
    }
    
    for( unsigned i=imin; i<imax+1; i++)//timestepping
    {
            start2d[0] = i;
            start2d_f[0] = i;
            start1d_f[0] = i;
            std::cout << "time = "<< time <<  std::endl;

            err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
            err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, npe[0].data());
            err = nc_inq_varid(ncid, names[1].data(), &dataIDs[1]);
            err = nc_get_vara_double( ncid, dataIDs[1], start2d, count2d, npe[1].data());
            err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
            err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d, phi.data());
            dg::blas1::transform(npe[0], npe[0], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
            dg::blas1::transform(npe[1], npe[1], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
            dg::blas1::transform( npe[0], logn[0], dg::LN<double>());
            dg::blas1::transform( npe[1], logn[1], dg::LN<double>());
            poisson.variationRHS(phi,uE);
            dg::blas1::pointwiseDot(npe[0],logn[0],energies[0]);
            dg::blas1::pointwiseDot(npe[1],logn[1],energies[1]);
            dg::blas1::pointwiseDot(npe[1],uE,energies[2]);
            dg::blas1::scal( energies[0], -p.tau[0]);
            dg::blas1::scal( energies[1], p.tau[1]);
            dg::blas1::scal( energies[2], 0.5*p.mu[1]);
	    dg::blas1::pointwiseDivide(npe[0],nprof,ntilde[0]);
	    dg::blas1::axpby(1.0,ntilde[0],-1.0,one,ntilde[0]);
            dg::blas1::pointwiseDot(one,ntilde[0],energies[3]);
            dg::blas1::pointwiseDot(phi,one,energies[4]);
           
            for (unsigned j=0;j<5;j++)
            {
	      //Backscatter to equidistant grid
                dg::blas2::gemv( equi, energies[j],energiesequi[j]);
              //Fill (x,y) matrix with values, normalise trafo
                for( size_t m = 0; m < rows; m++) {
                    for( size_t n = 0; n < cols; n++) {
			temp(m,n) = energiesequi[j][n+m*cols];
                    }
                }
                //compute 2d spectrum E(kx,ky)
                drt_dft.r2c_T( temp, comspec); //forward real to complex transform
                for( size_t m = 0; m < g2d_f.Nx(); m++) {
                    for( size_t n = 0; n < g2d_f.Ny(); n++) {
                        squspec(m,n) = sqrt(std::real(comspec(2*m,n)*std::conj(comspec(2*m,n))))/sqrt((2.*(double)cols)*(double)rows) ; //FACTOR 2 added
			//grow rate for phi spec
                        if (j==4) gammasquspec(m,n) =(squspec(m,n) - gammasquspec(m,n))/deltaT;
                    }
                } 
                //Write E(kx,ky) spectrum
                err2d_f = nc_put_vara_double( ncid2d_f, dataIDs2d_f[j],   start2d_f, count2d_f, squspec.getPtr()); 
                //compute (normalised) shell spectrum                   		
                for (size_t mn=0;mn<(unsigned)g2d_f.Nx();mn++) {
                    shellspec[mn]=0.;
		    counter[mn]=0;
                    for( size_t m = 0; m <g2d_f.Nx(); m++) {
                        for( size_t n = 0; n < g2d_f.Ny(); n++){			  
                            if((unsigned)(sqrt(m*m+n*n) - mn)<1) {
                                counter[mn]+=1;
			        shellspec[mn] += squspec(m,n);
				//grow rate for phi spec
                                if (j==4) gammashellspec[mn] =(shellspec[mn] - gammashellspec[mn]);
                            }
                        }
                    }
		    //normalise
		    shellspec[mn] /= counter[mn];
		    //grow rate for phi spec
		    if (j==4) gammashellspec[mn] /= (counter[mn]*deltaT);	
                }
                //Write E(k) spectrum
                err1d_f = nc_put_vara_double( ncid1d_f, dataIDs1d_f[j],   start1d_f, count1d_f, shellspec.data()); 
// 		todo                
                //compute E(ky) spectrum
                //compute E(kx) spectrum                
              }
            err2d_f = nc_put_vara_double( ncid2d_f, dataIDs2d_f[5],   start2d_f, count2d_f, gammasquspec.getPtr()); 
            err1d_f = nc_put_vara_double( ncid1d_f, dataIDs1d_f[5],   start1d_f, count1d_f, gammashellspec.data()); 

              
            err1d_f = nc_put_vara_double( ncid1d_f, dataIDs1d_f[6],   start1d_f, count1d_f, kn.data()); 
            err1d_f = nc_put_vara_double( ncid1d_f, tvarID1d_f, start1d_f, count1d_f, &time);
            err2d_f = nc_put_vara_double( ncid2d_f, tvarID2d_f, start2d_f, count2d_f, &time);
	    time += p.itstp*p.dt;

        
    }

    err = nc_close(ncid);

    err2d_f = nc_close(ncid2d_f);
    err1d_f = nc_close(ncid1d_f);

    return 0;
}

