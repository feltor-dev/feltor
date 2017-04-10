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

#include "dg/geometry.h"
#include "feltor/parameters.h"
#include "geometries/geometries.h"

int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.nc] \n";
        return -1;
    }
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    //////////////////////////////open nc file//////////////////////////////////
    file::NC_Error_Handle err;
    int ncidin;
    err = nc_open( argv[1], NC_NOWRITE, &ncidin);
    ///////////////////read in and show inputfile und geomfile//////////////////
    size_t length;
    err = nc_inq_attlen( ncidin, NC_GLOBAL, "inputfile", &length);
    std::string input( length, 'x');
    err = nc_get_att_text( ncidin, NC_GLOBAL, "inputfile", &input[0]);
    err = nc_inq_attlen( ncidin, NC_GLOBAL, "geomfile", &length);
    std::string geom( length, 'x');
    err = nc_get_att_text( ncidin, NC_GLOBAL, "geomfile", &geom[0]);
    err = nc_close(ncidin); 

    std::cout << "input "<<input<<std::endl;
    std::cout << "geome "<<geom <<std::endl;
    const eule::Parameters p(file::read_input( input));
    const dg::geo::solovev::GeomParameters gp(file::read_input( geom));
    p.display();
    gp.display();
    /////////////////////////////create Grids and weights//////////////////////////////////

    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Grids
    dg::CylindricalGrid3d<dg::DVec> g3d_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out, p.Nz_out, dg::NEU, dg::NEU, dg::PER);  
    dg::CartesianGrid2d  g2d_out( Rmin,Rmax, Zmin,Zmax, p.n_out, p.Nx_out, p.Ny_out, dg::NEU, dg::NEU);
    //--------generate nc file and define dimensions----------------
    int ncidout; 
    err = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncidout);
    err = nc_put_att_text( ncidout, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncidout, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int tvarID, dim_ids[4];
    err = file::define_dimensions( ncidout, dim_ids, &tvarID, g3d_out);

    //-----------Create 0d, 1d, 2d and 3d variables ----------------
    std::string names0d[5]  = {"correlationNphi", "Depsi_total", "fieldaligned", "com_vel", "midplane_mass"};
    std::string names1d[0]  = {};
    std::string names2d[0]  = {};
    std::string names3d[1]  = {"vorticity"};
    int dataIDs0d[5], dataIDs1d[0], dataIDs2d[0], dataIDs3d[1];
    for( unsigned i=0; i<5; i++)
        err = nc_def_var( ncidout, names0d[i].data(), NC_DOUBLE, 1, dim_ids, &dataIDs0d[i]);
    for( unsigned i=0; i<0; i++)
        err = nc_def_var( ncidout, names1d[i].data(), NC_DOUBLE, 2, dim_ids, &dataIDs1d[i]);
    for( unsigned i=0; i<0; i++)
        err = nc_def_var( ncidout, names2d[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs2d[i]);
    for( unsigned i=0; i<1; i++)
        err = nc_def_var( ncidout, names3d[i].data(), NC_DOUBLE, 4, dim_ids, &dataIDs3d[i]);
    err = nc_close(ncidout);
    ////////////////////////////Define input parameters ///////////////////////////
    std::string names[5] = {"electrons", "ions", "Ue", "Ui", "potential"}; 
    std::vector<dg::HVec> fields3d_h(5,dg::evaluate(dg::zero,g3d_out));
    std::vector<dg::DVec> fields3d_d(5,dg::evaluate(dg::zero,g3d_out));
    int dataIDs[5];
    size_t count[4]  = {1, g3d_out.Nz(), g3d_out.n()*g3d_out.Ny(), g3d_out.n()*g3d_out.Nx()};
    size_t start[4]  = {0, 0, 0, 0};
    double time=0.;

    ///////////////////////Define helper variables for computations////////
    dg::geo::solovev::MagneticField c(gp);
    dg::DDS dsN( typename dg::DDS::FieldAligned(
                dg::geo::Field<dg::geo::solovev::MagneticField>(c, gp.R_0), 
                g3d_out, gp.rk4eps, 
                dg::geo::PsiLimiter<dg::geo::solovev::Psip>(c.psip, gp.psipmaxlim), 
                dg::NEU,(2*M_PI)/((double)p.Nz)), 
            dg::geo::Field<dg::geo::solovev::MagneticField>(c, gp.R_0), dg::normed, dg::centered );
    dg::DVec vor3d    = dg::evaluate( dg::zero, g3d_out);
    dg::HVec transfer = dg::evaluate( dg::zero, g3d_out);
    dg::Elliptic<dg::CylindricalGrid3d<dg::DVec>, dg::DMatrix, dg::DVec> laplacian(g3d_out,dg::DIR, dg::DIR, dg::normed, dg::centered); 
    const dg::DVec w3d = dg::create::weights( g3d_out);   
    const dg::DVec w2d = dg::create::weights( g2d_out);   
    const dg::DVec curvR = dg::evaluate( dg::geo::CurvatureNablaBR<dg::geo::solovev::MagneticField>(c, gp.R_0), g3d_out);
    const dg::DVec curvZ = dg::evaluate( dg::geo::CurvatureNablaBZ<dg::geo::solovev::MagneticField>(c, gp.R_0), g3d_out);
    dg::Poisson<dg::CylindricalGrid3d<dg::DVec>, dg::DMatrix, dg::DVec> poisson(g3d_out,  dg::DIR, dg::DIR,  g3d_out.bcx(), g3d_out.bcy());
    const dg::DVec binv = dg::evaluate( dg::geo::Field<dg::geo::solovev::MagneticField>(c, gp.R_0) , g3d_out) ;
    const dg::DVec one3d    =  dg::evaluate(dg::one,g3d_out);
    const dg::DVec one2d    =  dg::evaluate(dg::one,g2d_out);
    dg::DVec temp1 = dg::evaluate(dg::zero , g3d_out) ;
    dg::DVec temp2 = dg::evaluate(dg::zero , g3d_out) ;
    dg::DVec npe = dg::evaluate(dg::zero , g3d_out) ; //y[0] + 1 = Ne

    //rewritten if straight fieldlines
    dg::DVec psipR =  dg::evaluate( dg::geo::solovev::PsipR(gp), g3d_out);
    dg::DVec psipZ =  dg::evaluate( dg::geo::solovev::PsipZ(gp), g3d_out);
    dg::blas1::pointwiseDot( psipR, psipR, temp1);
    dg::blas1::pointwiseDot( psipZ, psipZ, temp2);
    dg::blas1::axpby( 1., temp1, 1., temp2);
    dg::blas1::transform( temp2, temp2, dg::SQRT<double>());
    dg::DVec psipMag = temp2; //|\nabla \psi_p|
    bool straight = true;
    if( gp.A != 0 ) straight = false;
    for( unsigned i=0; i<gp.c.size(); i++)
        if( gp.c[i] != 0) straight = false;

    if( straight == true)
    {
        psipR = one3d;
        psipZ = dg::evaluate(dg::zero, g3d_out);
        psipMag = one3d;
    }
    dg::DVec Depsip3d =  dg::evaluate(dg::zero , g3d_out) ;   
    dg::DVec Depsip2d =  dg::evaluate(dg::zero , g2d_out) ;   

    std::vector<dg::DVec> fields2d(5,dg::evaluate(dg::zero,g2d_out));
    /////////////////////////begin timestepping/////////////////////////
    unsigned outlim = p.maxout;
    for( unsigned i=0; i<outlim; i++)//timestepping
    {
        err = nc_open(argv[2], NC_WRITE, &ncidout);
        start[0] = i; //set specific time  
        //write time data
        time += p.itstp*p.dt;
        err = nc_put_vara_double( ncidout, tvarID, start, count, &time);
        std::cout << "\rTimestep = " << i+1 << "/"<<outlim<<"  time = " << time << std::flush;
        //---------------READ: Ne,Ni,Ue,Ui,Phi ----------------------
        for( unsigned j=0;j<5; j++)
        {
            err = nc_open( argv[1], NC_NOWRITE, &ncidin); //open 3d file
            err = nc_inq_varid(ncidin, names[j].data(), &dataIDs[j]);
            err = nc_get_vara_double( ncidin, dataIDs[j], start, count, fields3d_h[j].data());
            fields3d_d[j] = fields3d_h[j];
            err = nc_close(ncidin);  //close 3d file
        }
        dg::blas1::transform( fields3d_d[0], npe, dg::PLUS<double>(+1));
        //get 2d density of MidPlane
        unsigned kmp = (g3d_out.Nz()/2);
        thrust::copy( fields3d_d[0].begin() + kmp*g2d_out.size(), fields3d_d[0].begin() + (kmp+1)*g2d_out.size(), fields2d[0].begin());
        //----------------Start vorticity computation
        dg::blas2::gemv( laplacian,fields3d_d[4],vor3d);
        dg::blas1::transfer( vor3d, transfer);
        err = nc_put_vara_double( ncidout, dataIDs3d[0], start, count, transfer.data());
        //----------------Stop vorticity computation
        //--------------- Start RADIALELECTRONDENSITYFLUX computation
        //ExB term  =  1/B[phi,psi_p] term
        dg::blas2::gemv( poisson.dxlhs(), fields3d_d[4], temp1); //temp1 = d_R phi
        dg::blas2::gemv( poisson.dylhs(), fields3d_d[4], temp2);  //temp2 = d_Z phi
        dg::blas1::pointwiseDot( psipZ, temp1, temp1);//temp1 = d_R phi d_Z psi_p
        dg::blas1::pointwiseDot( psipR, temp2, temp2); //temp2 = d_Z phi d_R psi_p 
        dg::blas1::axpby( 1.0, temp1, -1.0,temp2, Depsip3d);  //Depsip3d=[phi,psip]_RZ
        dg::blas1::pointwiseDot( Depsip3d, binv, Depsip3d); //Depsip3d = 1/B*[phi,psip]_RZ             
        //Curvature Term = (0.5*mu_e U_e^2-1) K(psi_p) term
        dg::blas1::pointwiseDot( curvR,  psipR, temp1);  //temp1 = K^R d_R psi
        dg::blas1::pointwiseDot( curvZ,  psipZ, temp2);  //temp2 = K^Z d_Z psi
        dg::blas1::axpby( 1.0, temp1, 1.0,temp2,  temp2);  //temp2 =K(psi_p)
        dg::blas1::pointwiseDot(fields3d_d[2], fields3d_d[2], temp1); // temp1=U_e^2
        dg::blas1::pointwiseDot(temp1,temp2, temp1); // temp1=U_e^2 K(psi_p)
        dg::blas1::axpby( -1.0, temp2,1.0,  Depsip3d );  //Depsip3d = 1/B*[phi,psi_p]_RZ - K(psi_p) 
        dg::blas1::axpby(  0.5*p.mu[0], temp1, 1.0,  Depsip3d);  //Depsip3d = 1/B*[phi,psi_p]_RZ - K(psi_p) + 0.5*nu_e*U_e^2*K(psi_p)
        dg::blas1::pointwiseDot( Depsip3d, npe, Depsip3d); //Depsip3d = N_e*(1/B*[phi,psi_p]_RZ - K(psi_p) + 0.5*nu_e*U_e^2*K(psi_p))
        dg::blas1::pointwiseDivide( Depsip3d, psipMag, Depsip3d); //Depsip3d = J\nabla\psi/|\nabla\psi|
        double blobMass = dg::blas2::dot( fields3d_d[0], w3d, one3d);
        double totalCurrent = dg::blas2::dot( Depsip3d, w3d, one3d)/blobMass;
        err = nc_put_vara_double( ncidout, dataIDs0d[1], start, count, &totalCurrent);
        thrust::copy( Depsip3d.begin() + kmp*g2d_out.size(), Depsip3d.begin() + (kmp+1)*g2d_out.size(), Depsip2d.begin()); //copy midplane
        double blob_mass_midplane = dg::blas2::dot( fields2d[0], w2d, one2d);
        err = nc_put_vara_double( ncidout, dataIDs0d[4], start, count, &blob_mass_midplane);
        double com = 1./blob_mass_midplane*dg::blas2::dot( Depsip2d, w2d, one2d);
        err = nc_put_vara_double(ncidout, dataIDs0d[3], start, count, &com);
        //------------------------STOP RADIALELECTRONDENSITYFLUX
        //------------------------Start NPhi Correlation computation
        dg::blas1::transform( fields3d_d[4], temp1, dg::EXP<double>());
        dg::blas1::transform( npe, temp2, dg::PLUS<double>(0));
        double norm1 = sqrt(dg::blas2::dot(temp1, w3d, temp1));
        double norm2 = sqrt(dg::blas2::dot(temp2, w3d, temp2));
        double correlation = dg::blas2::dot( temp1, w3d, temp2)/norm1/norm2;  //<phi, lnN>/||phi||/||lnN||
        err = nc_put_vara_double( ncidout, dataIDs0d[0], start, count, &correlation); 
        //--------------- Stop NPhi Correlation computation
        //----------------Start fieldaligned computation----------------
        dsN.centered( npe, temp1); //nabla_p N
        dg::blas1::pointwiseDivide( temp1, npe, temp2); //1/N nabla_p N
        double aligned = dg::blas2::dot( temp1, w3d, temp2);
        err = nc_put_vara_double( ncidout, dataIDs0d[2], start, count, &aligned);
        //----------------Stop fieldaligned computation-----------------
        err = nc_close(ncidout);  //close netcdf files
    } //end timestepping
    std::cout << std::endl;
    return 0;
}

