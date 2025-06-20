#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include "spectral/drt_dft.h"
// #include "spectral/drt_drt.h"

#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "feltorShw/parameters.h"
int main( int argc, char* argv[])
{
    if( argc != 4)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [outputkxky.nc] [outputk.nc]\n";
        return -1;
    }
//     std::ofstream os( argv[2]);
//     std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    //////////////////////////////open nc file//////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    ///////////////////read in and show inputfile//////////////////
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    std::cout << "input "<<input<<std::endl;
    auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
    const eule::Parameters p(js);
    p.display(std::cout);
    
    //////////////////////////////Grids//////////////////////////////////////
    //input grid
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    const size_t Ny =  g2d.n()*g2d.Ny();
    const size_t Nx =  g2d.n()*g2d.Nx();
    //output grids
    const double kxmin = 0./p.lx;
    const double kymin = 0./p.ly;    
    const unsigned Nkx = Nx; 
    const unsigned Nky = Ny/2+1;       
    const double kxmax = Nkx;//(Nkx/2.+1)/p.lx; //see fftw docu
    const double kymax = Nky;//(Nky)/p.ly; 
    const unsigned Nk = (unsigned)sqrt(Nkx*Nkx+Nky*Nky);
    const double kmin = sqrt(kxmin*kxmin + kymin*kymin);
    const double kmax = sqrt(kxmax*kxmax + kymax*kymax);
    
    //construct k-space grids    
    dg::Grid2d g2d_f( kxmin, kxmax, kymin, kymax, 1., Nkx, Nky, p.bc_x, p.bc_y);
    dg::Grid1d g1d_f( kmin, kmax,1., Nk,  p.bc_y);
    dg::Grid1d g1dx_f( kxmin, kxmax,1., Nkx,  p.bc_x);
    dg::Grid1d g1dy_f( kymin, kymax,1., Nky,  p.bc_y);
    
    //unsigned i_mode = 0;
    //unsigned j_mode = 1*p.sigma;

    //2d field netcdf vars of input.nc
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    int dataIDs[4];
    
    
    //2d field netcdf vars of outputkxky.nc
    dg::file::NC_Error_Handle err2d_f;
    int ncid2d_f,dim_ids2d_f[3],dataIDs2d_f[3], tvarID2d_f;
    std::string names2d_f[3] = {"S(ne)","S(phi)","gamma(phi)"};    

    size_t count2d_f[3]  = {1, g2d_f.Ny(), g2d_f.Nx()};
    size_t start2d_f[3]  = {0, 0, 0};    
    err2d_f = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid2d_f);
    err2d_f = nc_put_att_text( ncid2d_f, NC_GLOBAL, "inputfile", input.size(), input.data());
    err2d_f = dg::file::define_dimensions( ncid2d_f, dim_ids2d_f, &tvarID2d_f, g2d_f);
    for( unsigned i=0; i<3; i++){
        err2d_f = nc_def_var( ncid2d_f, names2d_f[i].data(), NC_DOUBLE, 3, dim_ids2d_f, &dataIDs2d_f[i]);
    }   
    err2d_f = nc_close(ncid2d_f); 
    
    //1d file netcdf vars of outputk.nc
    dg::file::NC_Error_Handle err1d_f;
    int ncid1d_f,dim_ids1d_f[2],dataIDs1d_f[4], tvarID1d_f;
    std::string names1d_f[4] = {"Sk(ne)","Sk(phi)","gamma(phi)","k",}; //may  goto ln(n/<n>)
    size_t count1d_f[2]  = {1, g1d_f.N()};
    size_t start1d_f[2]  = {0, 0};    
    err1d_f = nc_create(argv[3],NC_NETCDF4|NC_CLOBBER, &ncid1d_f);
    err1d_f = nc_put_att_text( ncid1d_f, NC_GLOBAL, "inputfile", input.size(), input.data());
    err1d_f = dg::file::define_dimensions( ncid1d_f, dim_ids1d_f, &tvarID1d_f, g1d_f);
    for( unsigned i=0; i<4; i++){
        err1d_f = nc_def_var( ncid1d_f, names1d_f[i].data(), NC_DOUBLE, 2, dim_ids1d_f, &dataIDs1d_f[i]);
    }   
    err1d_f = nc_close(ncid1d_f); 
    
    //dg stuff
    dg::HVec phi(dg::evaluate(dg::zero,g2d));
    dg::HVec one(dg::evaluate(dg::one,g2d));
    dg::HVec nprof(dg::evaluate(dg::ExpProfX(p.nprofileamp, p.bgprofamp,p.invkappa),g2d));
    std::vector<dg::HVec> npe(2,dg::evaluate(dg::zero,g2d));
    std::vector<dg::HVec> ntilde(2,dg::evaluate(dg::zero,g2d));    
    std::vector<dg::HVec> energies(2,phi); //Se,Si,SE
    std::vector<dg::HVec> energiesequi(2,phi); 
    dg::HVec k = dg::evaluate(dg::cooX1d,g1d_f);
    //scatter matrix
    dg::IHMatrix equi = dg::create::backscatter( g2d);
 
    //spectral stuff 
    spectral::Matrix<double, spectral::TL_DRT_DFT> tempxy(  Ny, Nx);
    spectral::Matrix<std::complex<double> > tempkykx(g2d_f.Nx(), g2d_f.Ny());  //# = (Nx, Ny/2+1)
    spectral::Matrix<double> kxkyspec( g2d_f.Ny(), g2d_f.Nx()); //Dimensions are transposed of tempkykx = ( Ny/2+1,Nx)
    spectral::Matrix<double> kxkyspec_old( g2d_f.Ny(), g2d_f.Nx()); //Dimensions are transposed of tempkykx = ( Ny/2+1,Nx)
    spectral::Matrix<double> gammakxkyspec( g2d_f.Ny(), g2d_f.Nx());
    std::vector<double> gammakspec( g1d_f.N());    
    std::vector<double> kspec( g1d_f.N());
    std::vector<unsigned> counter( g1d_f.N());
     
    //FFTW SETUP
    //FFTW_RODFT11 computes an RODFT11 transform, i.e. a DST-IV. (Logical N=2*n, inverse is FFTW_RODFT11.)  -> DIR_NEU
    //FFTW_RODFT10 computes an RODFT10 transform, i.e. a DST-II. (Logical N=2*n, inverse is FFTW_RODFT01.) -> DIR_DIR
    //FFTW_RODFT00 computes an RODFT00 transform, i.e. a DST-I. (Logical N=2*(n+1), inverse is FFTW_RODFT00.) -> DIR_DIR
//     fftw_r2r_kind kind = FFTW_RODFT10; //DFT & DST 2
    fftw_r2r_kind kind = FFTW_RODFT00; //DFT & DST 1
    spectral::DRT_DFT trafo( Ny, Nx, kind);
    
    //open netcdf files
    err2d_f = nc_open( argv[2], NC_WRITE, &ncid2d_f);
    err1d_f = nc_open( argv[3], NC_WRITE, &ncid1d_f);
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

    //initialize gammas
    for( unsigned m = 0; m < gammakxkyspec.rows(); m++) {
        for( unsigned n = 0; n < gammakxkyspec.cols(); n++) {
            gammakxkyspec(m,n) =0.;
        }
    }
    for (unsigned mn=0;mn<g1d_f.N();mn++) { 
        gammakspec[mn]=0.;
        kspec[mn]=0.;
    }


    for( unsigned i=imin; i<=imax; i++)//timestepping
    {
            start2d[0] = i;
            start2d_f[0] = i;
            start1d_f[0] = i;
//             std::cout << "time = "<< time << " i = " << i <<  std::endl;

            //get input.nc data
            err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]);
            err = nc_get_vara_double( ncid, dataIDs[0], start2d, count2d, npe[0].data());
            err = nc_inq_varid(ncid, names[1].data(), &dataIDs[1]);
            err = nc_get_vara_double( ncid, dataIDs[1], start2d, count2d, npe[1].data());
            err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
            err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d, phi.data());

            
            //compute tilde_N
	    if (p.modelmode==0 || p.modelmode==1)
	    {
	      dg::blas1::transform( npe[0], npe[0], dg::PLUS<>(p.bgprofamp + p.nprofileamp));
	      dg::blas1::transform( npe[1], npe[1], dg::PLUS<>(p.bgprofamp + p.nprofileamp));

	      dg::blas1::pointwiseDivide(npe[0],nprof,ntilde[0]);
	      dg::blas1::axpby(1.0,ntilde[0],-1.0,one,ntilde[0]);
	      dg::blas1::pointwiseDot(one,ntilde[0],energies[0]);
	    }
	    if (p.modelmode==2) {
	      dg::blas1::pointwiseDot(one,npe[0],energies[0]);
	    }
		
            dg::blas1::pointwiseDot(phi,one,energies[1]);


            for (unsigned j=0;j<2;j++)
            {
                //Backscatter to equidistant grid
                dg::blas2::gemv( equi, energies[j],energiesequi[j]);
                //Fill (x,y) matrix with values of xy data
                for( unsigned m = 0; m < tempxy.rows(); m++) {
                    for( unsigned n = 0; n <tempxy.cols(); n++) {   
                        tempxy(m,n) = energiesequi[j][n+m*tempxy.cols()];
                    }
                }
                //compute 2d spectrum E(kx,ky) with forward real to complex transform
                trafo.r2c_T( tempxy, tempkykx); //Note that output is transposed

                for( unsigned m = 0; m <  tempkykx.rows(); m++) {
                    for( unsigned n = 0; n < tempkykx.cols(); n++) {
                        //transpose absolute of transposed output
                        kxkyspec(n,m) = std::abs(tempkykx(m,n)); //is padded -> Y?
                        //normalise trafo

//                      kxkyspec(n,m)/=sqrt(2.*((double)Nx+1.)*(double)Ny); //for rodft00
//                      kxkyspec(n,m)/=sqrt(2.*(double)Nx*(double)Ny); //otherwise
			
                        //grow rate for phi spec with simple forward difference in time \gamma(kx,ky) = |ln \phi(kx,ky,(t+1))|-|ln \phi(kx,ky,(t))|/(\Delta t)
			if (j==1) {
			  if (i==0) { 
			      gammakxkyspec(n,m) = 0.;
			      kxkyspec_old(n,m)  = kxkyspec(n,m);
			  }
			  if (i>0) { 
			      gammakxkyspec(n,m) =(log(kxkyspec(n,m)) - log(kxkyspec_old(n,m)))/deltaT;
			      kxkyspec_old(n,m) = kxkyspec(n,m);
			  }
			}
                    }
                } 
                //Write E(kx,ky) spectrum
                err2d_f = nc_put_vara_double( ncid2d_f, dataIDs2d_f[j],   start2d_f, count2d_f, kxkyspec.getPtr()); 
                
//                 compute (normalised) shell spectrum        
                for (unsigned mn=0;mn<g1d_f.N();mn++) {
                    kspec[mn]=0.;
                    counter[mn]=0;
                    for( unsigned m = 0; m <kxkyspec.rows(); m++) {
                        for( unsigned n = 0; n < kxkyspec.cols(); n++){           
                            if((unsigned)(sqrt(m*m+n*n) - mn)<1) {
                                counter[mn]+=1;
                                kspec[mn] += kxkyspec(m,n);
                                //grow rate for phi spec
                                if (j==1) gammakspec[mn] =(kspec[mn] - gammakspec[mn]);
                            }
                        }
                    }
                    //normalise
                    kspec[mn] /= counter[mn];
                    //grow rate for phi spec
                    if (j==1) gammakspec[mn] /= (counter[mn]*deltaT);   
                }

                //Write E(k) spectrum
                err1d_f = nc_put_vara_double( ncid1d_f, dataIDs1d_f[j],   start1d_f, count1d_f, kspec.data()); 
                //      todo                
                //compute E(ky) spectrum
                //compute E(kx) spectrum                

              }

//             if (i>=2 ) gammakspecavg+=gammakxkyspec(j_mode,i_mode);
//  if (i==15)     std::cout << p.sigma << " " << gammakxkyspec(j_mode,i_mode)<< " " <<  p.invkappa <<" " <<  p.alpha<<" " <<  p.ly <<" "    
            err2d_f = nc_put_vara_double( ncid2d_f, dataIDs2d_f[2],   start2d_f, count2d_f, gammakxkyspec.getPtr()); 
            err1d_f = nc_put_vara_double( ncid1d_f, dataIDs1d_f[2],   start1d_f, count1d_f, gammakspec.data()); 
            err1d_f = nc_put_vara_double( ncid1d_f, dataIDs1d_f[3],   start1d_f, count1d_f, k.data()); 
            err1d_f = nc_put_vara_double( ncid1d_f, tvarID1d_f, start1d_f, count1d_f, &time);
            err2d_f = nc_put_vara_double( ncid2d_f, tvarID2d_f, start2d_f, count2d_f, &time);

            //advance time
            time += p.itstp*p.dt;        
    }

    err1d_f = nc_close(ncid1d_f);
    err2d_f = nc_close(ncid2d_f);
    err = nc_close(ncid);
    return 0;
}

