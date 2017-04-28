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

#include "file/nc_utilities.h"

#include "geometries/geometries.h"
#include "feltor/parameters.h"

// #define RADIALELECTRONDENSITYFLUX
// #define GRADIENTLENGTH
// #defina FSA


int main( int argc, char* argv[])
{
    if( argc != 4)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [outputfsa.nc] [output2d.nc]\n";
        return -1;
    }
//     std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<" & "<<argv[3]<<std::endl;

    //////////////////////////////open nc file//////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    ///////////////////read in and show inputfile und geomfile//////////////////
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    err = nc_inq_attlen( ncid, NC_GLOBAL, "geomfile", &length);
    std::string geom( length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "geomfile", &geom[0]);
    err = nc_close(ncid); 

    std::cout << "input "<<input<<std::endl;
    std::cout << "geome "<<geom <<std::endl;
    Json::Reader reader;
    Json::Value js,gs;
    reader.parse( input, js, false);
    const eule::Parameters p(js);
    reader.parse( geom, gs, false);
    const dg::geo::solovev::GeomParameters gp(gs);
    p.display();
    gp.display();
    ///////////////////////////////////////////////////////////////////////////

    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Grids

    dg::CylindricalGrid3d<dg::DVec> g3d_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out, p.Nz_out, p.bc, p.bc, dg::PER); 
    dg::Grid2d  g2d_out( Rmin,Rmax, Zmin,Zmax, p.n_out, p.Nx_out, p.Ny_out,  p.bc, p.bc); 
    //1d grid

    dg::geo::solovev::Psip psip(gp);
    dg::HVec transfer2d = dg::evaluate(dg::zero,g2d_out);
    dg::DVec w3d = dg::create::volume( g3d_out);   
        //Grids

    dg::DVec psipog2d   = dg::evaluate( psip, g2d_out);    
    double psipmin = (double)thrust::reduce( psipog2d.begin(), psipog2d.end(), 0.0,thrust::minimum<double>()  );
    double psipmax = (double)thrust::reduce( psipog2d.begin(), psipog2d.end(), psipmin,thrust::maximum<double>()  );
//     double psipmax = 0.0;
    dg::geo::PsiPupil<dg::geo::solovev::Psip> psipupil(psip,psipmax);
    dg::HVec psipupilog2d   = dg::evaluate( psipupil, g2d_out);    
    dg::HVec psipupilog3d   = dg::evaluate( psipupil, g3d_out);    

    unsigned Npsi = 50;//set number of psivalues
    std::cout << "psipmin =" << psipmin << " psipmax =" << psipmax << " Npsi =" << Npsi  <<std::endl;
    dg::Grid1d  g1d_out(psipmin  ,psipmax ,3, Npsi,dg::NEU); //one dimensional sipgrid
    dg::DVec w1d = dg::create::weights( g1d_out);   
    dg::HVec transfer1d = dg::evaluate(dg::zero,g1d_out);
    //read in midplane of electrons, ions Ue, Ui, and potential, and energy
    std::string names[5] = {"electrons", "ions", "Ue", "Ui", "potential"}; 
    int dataIDs[5];

    std::string names2d[13] = {"Ne_avg", "Ni_avg", "Ue_avg", "Ui_avg", "phi_avg","dNe_mp", "dNi_mp", "dUe_mp", "dUi_mp", "dphi_mp","vor_avg","Depsi_avg","Lperpinv_avg"}; 
    int dataIDs2d[13];
     //generate 2d nc file for one time step
    file::NC_Error_Handle err2d; 
    int ncid2d; 
    err2d = nc_create(argv[3],NC_NETCDF4|NC_CLOBBER, &ncid2d);
    err2d = nc_put_att_text( ncid2d, NC_GLOBAL, "inputfile", input.size(), input.data());
    err2d = nc_put_att_text( ncid2d, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[3], tvarID;
    err2d = file::define_dimensions( ncid2d, dim_ids, &tvarID, g2d_out);

    for( unsigned i=0; i<13; i++){
        err2d = nc_def_var( ncid2d, names2d[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs2d[i]);
    }   
    err2d = nc_close(ncid2d);

    //midplane 2d fields
    size_t count2d[3]  = {1, g3d_out.n()*g3d_out.Ny(), g3d_out.n()*g3d_out.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    size_t count3d[4]  = {1, g3d_out.Nz(), g3d_out.n()*g3d_out.Ny(), g3d_out.n()*g3d_out.Nx()};
    size_t start3d[4]  = {0, 0, 0, 0};
    //size_t count3dp[4] = {1, 1, g3d_out.n()*g3d_out.Ny(), g3d_out.n()*g3d_out.Nx()};
//     size_t start3dp[4] = {0, 0, 0, 0};

     //generate 1d nc file for one time step for the f(psi) quantities
    std::string names1d[10] = {"Ne_fsa", "Ni_fsa", "Ue_Fsa", "Ui_fsa", "phi_fsa","q","vor_fsa","Depsi_fsa","Lperpinv_fsa","psip1d"}; 
    int dataIDs1d[10];
    file::NC_Error_Handle err1d; 
    int ncid1d; 

    err1d = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid1d);
    err1d = nc_put_att_text( ncid1d, NC_GLOBAL, "inputfile", input.size(), input.data());
    err1d = nc_put_att_text( ncid1d, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids1d[2], tvarID1d;
    err1d = file::define_dimensions( ncid1d, dim_ids1d, &tvarID1d, g1d_out);
    for( unsigned i=0; i<10; i++){
        err1d = nc_def_var( ncid1d, names1d[i].data(), NC_DOUBLE, 2, dim_ids1d, &dataIDs1d[i]);
    }   
//  midplane 2d fields
    size_t count1d[2] = {1, g1d_out.n()*g1d_out.N()};
    size_t start1d[2] = {0, 0};

    err1d = nc_close(ncid1d);
    double time=0.;

//     double energy_0 =0.,U_i_0=0.,U_e_0=0.,U_phi_0=0.,U_pare_0=0.,U_pari_0=0.,mass_0=0.;
//     os << "#Time(1) mass(2) Ue(3) Ui(4) Uphi(5) Upare(6) Upari(7) Utot(8) EDiff(9)\n";
    std::cout << "Compute safety factor   "<< "\n";
    dg::geo::solovev::MagneticField c(gp);
    dg::geo::Alpha<dg::geo::solovev::MagneticField> alpha(c); 
    dg::DVec alphaog2d   = dg::evaluate( alpha, g2d_out);      
    dg::DVec abs = dg::evaluate( dg::cooX1d, g1d_out);
    dg::geo::SafetyFactor<dg::geo::solovev::MagneticField, dg::DVec> qprofile(g2d_out, gp, alphaog2d );
    dg::DVec sf = dg::evaluate(qprofile, g1d_out);


    //perp laplacian for computation of vorticity

    dg::DVec vor3d    = dg::evaluate( dg::zero, g3d_out);
    dg::Elliptic<dg::CylindricalGrid3d<dg::DVec>, dg::DMatrix, dg::DVec> laplacian(g3d_out,dg::DIR, dg::DIR, dg::normed, dg::centered); 
    dg::IDMatrix fsaonrzmatrix,fsaonrzphimatrix;     
    fsaonrzmatrix    =  dg::create::interpolation(psipupilog2d ,g1d_out);    
    fsaonrzphimatrix =  dg::create::interpolation(psipupilog3d ,g1d_out);    
    
    //Vectors and Matrices for Diffusion coefficient

    const dg::DVec curvR = dg::evaluate( dg::geo::CurvatureNablaBR<dg::geo::solovev::MagneticField>(c, gp.R_0), g3d_out);
    const dg::DVec curvZ = dg::evaluate( dg::geo::CurvatureNablaBZ<dg::geo::solovev::MagneticField>(c, gp.R_0), g3d_out);
    dg::Poisson<dg::CylindricalGrid3d<dg::DVec>,dg::DMatrix, dg::DVec> poisson(g3d_out,  dg::DIR, dg::DIR,  g3d_out.bcx(), g3d_out.bcy());
    const dg::DVec binv = dg::evaluate( dg::geo::Field<dg::geo::solovev::MagneticField>(c, gp.R_0) , g3d_out) ;
    dg::DVec temp1 = dg::evaluate(dg::zero , g3d_out) ;
    dg::DVec temp2 = dg::evaluate(dg::zero , g3d_out) ;
    dg::DVec temp3 = dg::evaluate(dg::zero , g3d_out) ;
    #ifdef RADIALELECTRONDENSITYFLUX
    const dg::DVec psipR =  dg::evaluate( dg::geo::solovev::PsipR(gp), g3d_out);
    const dg::DVec psipRR = dg::evaluate( dg::geo::solovev::PsipRR(gp), g3d_out);
    const dg::DVec psipZ =  dg::evaluate( dg::geo::solovev::PsipZ(gp), g3d_out);
    const dg::DVec psipZZ = dg::evaluate( dg::geo::solovev::PsipZZ(gp), g3d_out);
    const dg::DVec psipRZ = dg::evaluate( dg::geo::solovev::PsipRZ(gp), g3d_out);
    dg::DVec Depsip3d =  dg::evaluate(dg::zero , g3d_out) ;   
    dg::DVec one3d    =  dg::evaluate(dg::one,g3d_out);
    dg::DVec one1d    =  dg::evaluate(dg::one,g1d_out);
    #endif
#ifdef GRADIENTLENGTH
    dg::DVec Lperpinv3d =  dg::evaluate(dg::zero , g3d_out) ;   
#endif
    std::vector<dg::HVec> fields3d_h(5,dg::evaluate(dg::zero,g3d_out));
    std::vector<dg::DVec> fields3d(5,dg::evaluate(dg::zero,g3d_out));
    std::vector<dg::DVec> fields2d(5,dg::evaluate(dg::zero,g3d_out));
    dg::ToroidalAverage<dg::DVec> toravg(g3d_out);
    unsigned outlim = 0.; int timeID;
    size_t steps;
    err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
    err = nc_inq_unlimdim( ncid, &timeID);
    err = nc_inq_dimlen( ncid, timeID, &steps);
    steps-=1;
    outlim = steps/p.itstp;
    for( unsigned i=0; i<outlim; i++)//timestepping
    {
//      start3dp[0] = i; //set specific time  
        start3d[0] = i; //set specific time  
        start2d[0] = i;
        start1d[0] = i;
        time += p.itstp*p.dt;
        err2d = nc_open(argv[3], NC_WRITE, &ncid2d);
        err1d = nc_open(argv[2], NC_WRITE, &ncid1d);

        std::cout << "Timestep = " << i << "  time = " << time << "\n";

        //Compute toroidal average and fluctuation at midplane for every timestep
        dg::DVec data2davg = dg::evaluate( dg::zero, g2d_out);   
        dg::DVec data2dfsa = dg::evaluate( dg::zero, g2d_out);    
        dg::DVec vor2davg = dg::evaluate( dg::zero, g2d_out);
        dg::DVec Depsip2davg =  dg::evaluate(dg::zero , g2d_out); 
        dg::DVec Depsip3dfluc =  dg::evaluate(dg::zero , g3d_out);
        dg::DVec Depsip2dflucavg =  dg::evaluate(dg::zero , g2d_out);  
        dg::DVec Lperpinv2davg =  dg::evaluate(dg::zero , g2d_out);          
        //Ne,Ni,Ue,Ui,Phi
        for( unsigned j=0;j<5; j++)
        {
            //set quantities to zero
            data2davg = dg::evaluate( dg::zero, g2d_out);   
            data2dfsa = dg::evaluate( dg::zero, g2d_out);    

            //get 3d data
            err = nc_open( argv[1], NC_NOWRITE, &ncid); //open 3d file
            err = nc_inq_varid(ncid, names[j].data(), &dataIDs[j]);
            err = nc_get_vara_double( ncid, dataIDs[j], start3d, count3d, fields3d_h[j].data());
            err = nc_close(ncid);  //close 3d file
            fields3d[j] = fields3d_h[j];
    
            //get 2d data and sum up for avg
            toravg(fields3d[j],data2davg);

            //get 2d data of MidPlane
            unsigned kmp = (g3d_out.Nz()/2);
            dg::DVec data2dflucmid(fields3d[j].begin() + kmp*g2d_out.size(),fields3d[j].begin() + (kmp+1)*g2d_out.size());
            
            //for fluctuations to be  f_varphi
//             dg::blas1::axpby(1.0,data2dflucmid,-1.0,data2davg,data2dflucmid); //Compute z fluctuation
            dg::blas1::transfer(data2davg,transfer2d);            
            err2d = nc_put_vara_double( ncid2d, dataIDs2d[j],   start2d, count2d, transfer2d.data()); //write avg


            //computa fsa of quantities
            dg::geo::FluxSurfaceAverage<dg::geo::solovev::MagneticField, dg::DVec> fsadata(g2d_out,c, data2davg );
            dg::DVec data1dfsa = dg::evaluate(fsadata,g1d_out);
            dg::blas1::transfer(data1dfsa,transfer1d);
            err1d = nc_put_vara_double( ncid1d, dataIDs1d[j], start1d, count1d,  transfer1d.data());
            
            //compute delta f on midplane : df = f_mp - <f>
            dg::blas2::gemv(fsaonrzmatrix, data1dfsa, data2dfsa); //fsa on RZ grid
            dg::blas1::axpby(1.0,data2dflucmid,-1.0,data2dfsa,data2dflucmid); 
            dg::blas1::transfer(data2dflucmid,transfer2d);     
            err2d = nc_put_vara_double( ncid2d, dataIDs2d[j+5], start2d, count2d, transfer2d.data());

        }
        //----------------Start vorticity computation
        dg::blas2::gemv( laplacian,fields3d[4],vor3d);
        toravg(vor3d,vor2davg);
        dg::blas1::transfer(vor2davg,transfer2d);     

        err2d = nc_put_vara_double( ncid2d, dataIDs2d[10],   start2d, count2d, transfer2d.data());
        dg::geo::FluxSurfaceAverage<dg::geo::solovev::MagneticField, dg::DVec> fsavor(g2d_out,c, vor2davg );
        dg::DVec vor1dfsa = dg::evaluate(fsavor,g1d_out);
        dg::blas1::transfer(vor1dfsa,transfer1d);
        err1d = nc_put_vara_double( ncid1d, dataIDs1d[6], start1d, count1d,  transfer1d.data()); 
        //----------------Stop vorticity computation
        
        //--------------- Start RADIALELECTRONDENSITYFLUX computation
        dg::blas1::transform(fields3d[0], temp3, dg::PLUS<>(+1)); 
        #ifdef RADIALELECTRONDENSITYFLUX
        //ExB term  =  1/B[phi,psi_p] term
        dg::blas2::gemv( poisson.dxlhs(), fields3d[4], temp1); //temp1 = d_R phi
        dg::blas2::gemv( poisson.dylhs(), fields3d[4], temp2);  //temp2 = d_Z phi
        dg::blas1::pointwiseDot( psipZ, temp1, temp1);//temp1 = d_R phi d_Z psi_p
        dg::blas1::pointwiseDot( psipR, temp2, temp2); //temp2 = d_Z phi d_R psi_p 
        dg::blas1::axpby( 1.0, temp1, -1.0,temp2, Depsip3d);  //Depsip3d=[phi,psip]_RZ
        dg::blas1::pointwiseDot( Depsip3d, binv, Depsip3d); //Depsip3d = 1/B*[phi,psip]_RZ             
        //Curvature Term = -(1-0.5*mu_e U_e^2) K(psi_p) term
        dg::blas1::pointwiseDot( curvR,  psipR, temp1);  //temp1 = K^R d_R psi
        dg::blas1::pointwiseDot( curvZ,  psipZ, temp2);  //temp2 = K^Z d_Z psi
        dg::blas1::axpby( 1.0, temp1, 1.0,temp2,  temp2);  //temp2 =K(psi_p)
        dg::blas1::pointwiseDot(fields3d[2], fields3d[2], temp1); // temp1=U_e^2
        dg::blas1::pointwiseDot(temp1,temp2, temp1); // temp1=U_e^2 K(psi_p)
        dg::blas1::axpby( -1.0, temp2,1.0,  Depsip3d );  //Depsip3d = 1/B*[phi,psi_p]_RZ - K(psi_p) 
        dg::blas1::axpby(  0.5*p.mu[0], temp1, 1.0,  Depsip3d);  //Depsip3d = 1/B*[phi,psi_p]_RZ - K(psi_p) + 0.5*nu_e*U_e^2*K(psi_p)
        dg::blas1::pointwiseDot( Depsip3d, temp3, Depsip3d); //Depsip3d = N_e*(1/B*[phi,psi_p]_RZ - K(psi_p) + 0.5*nu_e*U_e^2*K(psi_p))
        
        //normalize by 1/|nabla psip|
        dg::blas1::pointwiseDot( psipR, psipR, temp1); // psipR^2
        dg::blas1::pointwiseDot( psipZ, psipZ, temp2); // psipZ^2
        dg::blas1::axpby(  1.0, temp1, 1.0,temp2,  temp1);  // psipR^2 +   psipZ^2
        dg::blas1::transform(temp1, temp1, dg::SQRT<double>());  // sqrt(psipR^2 +   psipZ^2)
        dg::blas1::pointwiseDivide( Depsip3d, temp1, Depsip3d); //Depsip3d = N_e*(1/B*[phi,psi_p]_RZ - K(psi_p) + 0.5*nu_e*U_e^2*K(psi_p))
      
        toravg(Depsip3d,Depsip2davg);

        dg::geo::FluxSurfaceAverage<dg::geo::solovev::MagneticField, dg::DVec> fsaDepsip(g2d_out,c, Depsip2davg );
        dg::DVec  Depsip1Dfsa = dg::evaluate(fsaDepsip,g1d_out);
        //compute delta f on midplane : d Depsip2d = Depsip - <Depsip>       
        dg::blas2::gemv(fsaonrzphimatrix, Depsip1Dfsa , Depsip3dfluc ); //fsa on RZ grid
        dg::blas1::axpby(1.0,Depsip3d,-1.0, Depsip3dfluc, Depsip3dfluc); 
        //Same procedure for fluc
        toravg(Depsip3dfluc,Depsip2dflucavg);
        //fluctuation
//         transfer2d = Depsip2dflucavg;
        //toroidal avg
        transfer2d = Depsip2davg;
        err2d = nc_put_vara_double( ncid2d, dataIDs2d[11],   start2d, count2d, transfer2d.data());
        dg::geo::FluxSurfaceAverage<dg::geo::solovev::MagneticField, dg::DVec> fsaDepsipfluc(g2d_out,c, Depsip2dflucavg );
        dg::DVec  Depsip1Dflucfsa = dg::evaluate(fsaDepsipfluc,g1d_out);
        transfer1d =Depsip1Dflucfsa;
        err1d = nc_put_vara_double( ncid1d, dataIDs1d[7], start1d, count1d,   transfer1d.data()); 
//         std::cout << "Depsip =" << dg::blas2::dot(psipupilog3d,w3d, Depsip3dfluc) << std::endl;
        #endif
        //STOP RADIALELECTRONDENSITYFLUX
        #ifdef GRADIENTLENGTH
        dg::blas1::transform(temp3, temp1, dg::LN<double>()); // lnN
        poisson.variationRHS(temp1,temp2); // (nabla_perp N)^2
        dg::blas1::transform(temp2, Lperpinv3d, dg::SQRT<double>()); // |(nabla_perp N)|
        toravg(Lperpinv3d,Lperpinv2davg);
        transfer2d = Lperpinv2davg;
        err2d = nc_put_vara_double( ncid2d, dataIDs2d[12],   start2d, count2d, transfer2d.data());
        dg::geo::FluxSurfaceAverage<dg::geo::solovev::MagneticField, dg::DVec> fsaLperpinv(g2d_out,c, Lperpinv2davg );
        dg::DVec  Lperpinv1Dfsa = dg::evaluate(fsaLperpinv,g1d_out);
        transfer1d =Lperpinv1Dfsa;
        err1d = nc_put_vara_double( ncid1d, dataIDs1d[8], start1d, count1d,   transfer1d.data()); 
//         std::cout << "Lperpinv=" <<dg::blas2::dot(psipupilog3d,w3d, Lperpinv3d) << std::endl;
        #endif
        
        //put safety factor into file
        dg::blas1::transfer(sf,transfer1d);
        err1d = nc_put_vara_double( ncid1d, dataIDs1d[5], start1d, count1d,  transfer1d.data());
        dg::blas1::transfer(abs,transfer1d);
        err1d = nc_put_vara_double( ncid1d, dataIDs1d[9], start1d, count1d, transfer1d.data());
        //write time data
        err1d = nc_put_vara_double( ncid1d, tvarID1d, start1d, count1d, &time);
        err1d = nc_close(ncid1d);  //close 1d netcdf files
        err2d = nc_put_vara_double( ncid2d, tvarID, start2d, count2d, &time);
        err2d = nc_close(ncid2d); //close 2d netcdf files
      
//         //Probe 
//         const dg::DVec Rprobe(1,gp.R_0+p.boxscaleRm*gp.a*0.8);
//         const dg::DVec Zprobe(1,0.0);
//         const dg::DVec Phiprobe(1,M_PI);
//         dg::IDMatrix probeinterp  = dg::create::interpolation( Rprobe,  Zprobe, Phiprobe, g3d_out, dg::NEU);
//         dg::DVec probevalue(1,0.0);
//         dg::blas2::gemv(probeinterp,fields3d[0],probevalue);
//         std::cout << probevalue[0]<< std::endl;
        // ---- Compute energies ----
//         std::cout << "Compute macroscopic timedependent quantities"<< "\n";

    

        //write macroscopic timedependent quantities into output.dat file
//         os << time << " " << mass_norm << " " <<  U_e_norm <<" " <<  U_i_norm <<" " << U_phi_norm <<" " << U_pare_norm <<" " << U_pari_norm <<" "  << energy_norm <<" " << energy_diff<<std::endl;
        
        
    } //end timestepping
    //cross coherence between phi and ne
    //relative fluctuation amplitude(R,Z,phi) = delta n(R,Z,phi)/n0(psi)
    
    return 0;
}

