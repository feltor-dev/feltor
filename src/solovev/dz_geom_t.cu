
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>

#include <cusp/print.h>

// #define DG_DEBUG


#include "file/read_input.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/xspacelib.cuh"
#include "geometry.h"
#include "init.h"
#include "dg/backend/dz.cuh"
#include "dg/algorithm.h"
#include "dg/backend/functions.h"
#include "dg/backend/interpolation.cuh"
#include "draw/host_window.h"
#include "geom_parameters.h"
#include "file/nc_utilities.h"
struct InvNormR
{
    InvNormR( solovev::GeomParameters gp): R_0(gp.R_0){}
    double operator()( double R, double Z, double phi)
    {
        return R_0/R;
    }
    
    private:
    double R_0;
}; 

int main( int argc, char* argv[])
{
    double Rmin,Zmin,Rmax,Zmax;
//     double A,R_0,a, elongation;
//     double psipmin,psipmax;
    unsigned n, Nx, Ny, Nz;

    std::vector<double> c(13);
    //read and store geom data
       std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params.txt"); 
            v2 = file::read_input( "../feltor/input.txt");

        }
        else
        {
            v = file::read_input( argv[1]); 
            v2 = file::read_input( argv[2]);
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}

    //write parameters from file into variables
    const solovev::GeomParameters gp(v);
    const eule::Parameters p(v2);
    p.display( std::cout);
    gp.display( std::cout);

    Rmin=gp.R_0-(p.boxscale)*gp.a;
    Zmin=-(p.boxscale)*gp.a*gp.elongation;
    Rmax=gp.R_0+(p.boxscale)*gp.a; 
    Zmax=(p.boxscale)*gp.a*gp.elongation;
    std::cout << "The grid parameters" <<"\n";
    std::cout  << Rmin<<" rho_s " << Rmax <<" rho_s " << Zmin <<" rho_s " <<Zmax <<" rho_s " <<"\n";
    std::cout << "Type n, Nx, Ny, Nz\n";
    std::cin >> n>> Nx>>Ny>>Nz;
        
    solovev::Field field(gp);
    solovev::Psip psip(gp.R_0,gp.A,gp.c);
    solovev::PsipR psipR(gp.R_0,gp.A,gp.c);
    solovev::PsipRR psipRR(gp.R_0,gp.A,gp.c);  
    solovev::PsipZ psipZ(gp.R_0,gp.A,gp.c);  
    solovev::PsipZZ psipZZ(gp.R_0,gp.A,gp.c);   
    solovev::PsipRZ psipRZ(gp.R_0,gp.A,gp.c);  
    solovev::Ipol ipol(gp.R_0,gp.A,psip);
    solovev::InvB invB(gp.R_0,ipol,psipR,psipZ);
    solovev::LnB lnB(gp.R_0,ipol,psipR,psipZ);
    solovev::BR bR(gp.R_0,gp.A,psipR,psipRR,psipZ,psipRZ,invB);
    solovev::BZ bZ(gp.R_0,gp.A,psipR,psipZ,psipZZ,psipRZ,invB);
    solovev::CurvatureR curvatureR(gp);
    solovev::CurvatureZ curvatureZ(gp);
    solovev::GradLnB gradLnB(gp);
    solovev::Pupil pupil(gp);
    InvNormR invnormr(gp);
    solovev::FieldR fieldR(gp);
    solovev::FieldZ fieldZ(gp);
    solovev::FieldP fieldP(gp);

    dg::Grid3d<double> grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,n, Nx, Ny,Nz);
    dg::HVec vecR = dg::evaluate( fieldR, grid);
    dg::HVec vecZ = dg::evaluate( fieldZ, grid);
    dg::HVec vecP = dg::evaluate( fieldP, grid);

    file::NC_Error_Handle err;
    int ncid, dim_ids[3];
//     err = nc_create( "geometry.nc", NC_CLOBBER, &ncid);
    err = nc_create( "geometry.nc", NC_NETCDF4|NC_CLOBBER, &ncid);

    err = file::define_dimensions( ncid, dim_ids, grid);
    int vecID[3];
    err = nc_def_var( ncid, "BR", NC_DOUBLE, 3, dim_ids, &vecID[0]);
    err = nc_def_var( ncid, "BZ", NC_DOUBLE, 3, dim_ids, &vecID[1]);
    err = nc_def_var( ncid, "BP", NC_DOUBLE, 3, dim_ids, &vecID[2]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, vecID[0], vecR.data());
    err = nc_put_var_double( ncid, vecID[1], vecZ.data());
    err = nc_put_var_double( ncid, vecID[2], vecP.data());
    nc_close(ncid);
    std::cout << "Check single field by integrating from 0 to 2pi" << "\n";
    dg::HVec v5(1, 0);
    std::vector<thrust::host_vector<double> > in(3, v5);
    std::vector<thrust::host_vector<double> > out(3, v5);
    in[0][0]=gp.R_0+0.9*gp.a; 
//     in[1][0]=0.9*gp.a*gp.elongation;
    in[1][0]=0.0;
    in[2][0]=0.;
    

    dg::integrateRK4( field, in, out,  2*M_PI, gp.rk4eps);
    
    std::cout <<"Rin =  "<< in[0][0] <<" Zin =  "<<in[1][0] <<" sin  = "<<in[2][0]<<"\n";
    std::cout <<"Rout = "<< out[0][0]<<" Zout = "<<out[1][0]<<" sout = "<<out[2][0]<<"\n";


    
    for (unsigned k=1;k<2;k++) //n iterator
    {
        for (unsigned i=0;i<1;i++) //Nxy iterator
        {
            std::stringstream ss1,ss2;
            ss1 << "dzerr1n" <<k*n<<"Nxy"<<pow(2,i)* Nx<<".txt";
            ss2 << "dzerr2n" <<k*n<<"Nxy"<<pow(2,i)* Nx<<".txt";
            std::string dzerr1fn = ss1.str();
            std::string dzerr2fn = ss2.str();
    //         std::cout << dzerr1fn;
            std::ofstream dzerrfile1((char *) dzerr1fn.c_str());
            std::ofstream dzerrfile2((char *) dzerr2fn.c_str());
            for (unsigned zz=0;zz<1;zz++) //Nz iterator
            {
                std::cout << "n = " << k*n << " Nx = " <<pow(2,i)* Nx << " Ny = " <<pow(2,i)* Ny << " Nz = "<<pow(2,zz)* Nz <<"\n";
                //Similar to feltor grid
                dg::Grid3d<double> g3d( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,k*n,pow(2,i)* Nx,pow(2,i)* Ny, pow(2,zz)*Nz,dg::DIR, dg::DIR, dg::PER, dg::cylindrical);
                const dg::DVec w3d = dg::create::weights( g3d);
                dg::DVec pupilongrid = dg::evaluate( pupil, g3d);


                std::cout <<"-----(1) test with testfunction" << "\n";
                solovev::TestFunction func(psip);
                solovev::DeriTestFunction derifunc(gp,psip,psipR,psipZ,ipol,invB);
                std::cout << "Construct parallel  derivative\n";
                dg::Timer t;
                t.tic();
                dg::DZ<dg::DMatrix, dg::DVec> dz( field, g3d,gp.rk4eps,solovev::PsiLimiter(gp)); 
                t.toc();
                std::cout << "Creation of parallel Derivative took "<<t.diff()<<"s\n";

                dg::DVec function = dg::evaluate( func, g3d),dzfunc(function);
                dg::DVec diff(g3d.size());

                dg::DVec solution = dg::evaluate( derifunc, g3d);
                dz( function, dzfunc);
                //cut boundaries
                dg::blas1::pointwiseDot( pupilongrid, dzfunc, dzfunc); 
                dg::blas1::pointwiseDot( pupilongrid, solution, solution); 

                dg::blas1::axpby( 1., solution, -1., dzfunc,diff);
                double normdz = dg::blas2::dot( w3d, dzfunc);
                std::cout << "Norm dz  = "<<sqrt( normdz)<<"\n";
                double normsol = dg::blas2::dot( w3d,solution);
                std::cout << "Norm sol = "<<sqrt( normsol)<<"\n";
                double normdiff = dg::blas2::dot( w3d, diff);
                double reldiff=sqrt( normdiff/normsol );
                std::cout << "Rel Diff = "<< reldiff<<"\n";

                std::cout <<"-----(2) test with gradlnb" << "\n";    
                dg::DVec gradLnBsolution = dg::evaluate( gradLnB, g3d);
                dg::DVec lnBongrid = dg::evaluate( lnB, g3d);
                dg::DVec dzlnBongrid(g3d.size());
                dg::DVec diff2(g3d.size());
                dz(lnBongrid,dzlnBongrid);
                
                //cut boundaries
                dg::blas1::pointwiseDot( pupilongrid, dzlnBongrid, dzlnBongrid); 
                dg::blas1::pointwiseDot( pupilongrid, gradLnBsolution, gradLnBsolution); 


                dg::blas1::axpby( 1., gradLnBsolution , -1., dzlnBongrid,diff2); //diff = gradlnB - dz(ln(B))
                dg::blas1::pointwiseDot( pupilongrid,diff2,diff2); 

                double normdz2 = dg::blas2::dot( w3d, dzlnBongrid); //=  Integral (gdz(ln(B))^2 )
                std::cout << "Norm dz  = "<<sqrt( normdz2)<<"\n";
                double normsol2 = dg::blas2::dot( w3d,gradLnBsolution);//=  Integral (gradlnB^2 )
                std::cout << "Norm sol = "<<sqrt( normsol2)<<"\n";
                double normdiff2=dg::blas2::dot( w3d, diff2); //=  Integral ((gradlnB - dz(ln(B)))^2)
                double reldiff2 =sqrt( normdiff2/normsol2 ); ;//=  sqrt(Integral ((gradlnB - dz(ln(B)))^2)/Integral (gradlnB^2 ))
                std::cout << "Rel Diff = "<<reldiff2 <<"\n";
                
                std::cout <<"-----(3) test with gradlnb and with Arakawa discretization" << "\n";    
                dg::ArakawaX< dg::DMatrix, dg::DVec>    arakawa(g3d); 
                dg::DVec invBongrid = dg::evaluate( invB, g3d);
                dg::DVec psipongrid = dg::evaluate( psip, g3d);
                dg::DVec invnormrongrid = dg::evaluate( invnormr, g3d);
                dg::DVec arakawasolution(g3d.size());
                dg::DVec diff3(g3d.size());

                arakawa( lnBongrid, psipongrid, arakawasolution); //1/B [B,psip]
                dg::blas1::pointwiseDot( invBongrid, arakawasolution, arakawasolution); 
                dg::blas1::pointwiseDot( invnormrongrid, arakawasolution, arakawasolution); 
                dg::blas1::pointwiseDot( pupilongrid, arakawasolution, arakawasolution); 
                dg::blas1::axpby( 1., gradLnBsolution , -1., arakawasolution,diff3);
                double normarak= dg::blas2::dot( w3d, arakawasolution); //=  Integral (gdz(ln(B))^2 )
                std::cout << "Norm normarak  = "<<sqrt( normarak)<<"\n";
                double normdiff3=dg::blas2::dot( w3d, diff3); //=  Integral ((gradlnB - dz(ln(B)))^2)
                double reldiff3 =sqrt( normdiff3/normsol2 ); ;//=  sqrt(Integral ((gradlnB - dz(ln(B)))^2)/Integral (gradlnB^2 ))
                std::cout << "Rel Diff = "<<reldiff3 <<"\n";
                dzerrfile1 << pow(2,zz)*Nz <<" " << reldiff << std::endl;
                dzerrfile2 << pow(2,zz)*Nz <<" " << reldiff2 << std::endl;
                
                std::cout <<"----(4) test div(B) != 0 "<<"\n";
                dg::DVec bRongrid = dg::evaluate( fieldR, grid);
                dg::DVec bZongrid = dg::evaluate( fieldZ, grid);
                dg::DVec dRbR(g3d.size());
                dg::DVec dZbZ(g3d.size());                
                dg::DVec invRbR(g3d.size());                
//                    //cut boundaries
                dg::blas1::pointwiseDot( pupilongrid, bRongrid,bRongrid); 
                dg::blas1::pointwiseDot( pupilongrid, bZongrid, bZongrid); 
                
                dg::DVec divB(g3d.size());                
                dg::blas2::gemv( arakawa.dx(), bRongrid, dRbR);
                dg::blas2::gemv( arakawa.dy(), bZongrid, dZbZ);
                dg::blas1::pointwiseDot( invnormrongrid , bRongrid, invRbR);
                dg::blas1::axpby( 1., dRbR   , 1., dZbZ, divB);
                dg::blas1::axpby( 1./gp.R_0, invRbR , 1., divB);
                double normdivB2= dg::blas2::dot( w3d, divB); 
                std::cout << "divB = "<<sqrt( normdivB2)<<"\n";
                
             }
            dzerrfile1.close();
            dzerrfile2.close();
   
        }

    }
    

    return 0;
}
