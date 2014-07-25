
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>

#include <cusp/print.h>

#define DG_DEBUG

#include "file/read_input.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/xspacelib.cuh"
#include "geometry.cuh"
#include "dg/backend/dz.cuh"
#include "dg/algorithm.h"
#include "dg/backend/functions.h"
#include "dg/backend/interpolation.cuh"
#include "draw/host_window.h"


int main()
{
    double Rmin,Zmin,Rmax,Zmax;
//     double A,R_0,a, elongation;
//     double psipmin,psipmax;
    unsigned n, Nx, Ny, Nz;

    std::vector<double> c(13);
    //read and store geom data
    std::vector<double> v;
    try{ v = file::read_input( "geometry_params.txt"); }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    for( unsigned i = 1; i<v.size(); i++)
    std::stringstream title;
    //write parameters from file into variables
    const dg::GeomParameters gp(v);
    gp.display( std::cout);

    Rmin=gp.R_0-1.1*gp.a;
    Zmin=-1.1*gp.a*gp.elongation;
    Rmax=gp.R_0+1.1*gp.a; 
    Zmax=1.1*gp.a*gp.elongation;
    std::cout << "The grid parameters" <<"\n";
    std::cout  << Rmin<<"rho_s " << Rmax <<"rho_s " << Zmin <<"rho_s " <<Zmax <<"rho_s " <<"\n";
    std::cout << "Type n, Nx, Ny, Nz\n";
    std::cin >> n>> Nx>>Ny>>Nz;
        dg::Field field(gp);
    dg::Psip psip(gp.R_0,gp.A,gp.c);
    dg::PsipR psipR(gp.R_0,gp.A,gp.c);
    dg::PsipRR psipRR(gp.R_0,gp.A,gp.c);  
    dg::PsipZ psipZ(gp.R_0,gp.A,gp.c);  
    dg::PsipZZ psipZZ(gp.R_0,gp.A,gp.c);   
    dg::PsipRZ psipRZ(gp.R_0,gp.A,gp.c);  
    dg::Ipol ipol(gp.R_0,gp.A,psip);
    dg::InvB invB(gp.R_0,ipol,psipR,psipZ);
    dg::LnB lnB(gp.R_0,ipol,psipR,psipZ);
    dg::BR bR(gp.R_0,gp.A,psipR,psipRR,psipZ,psipRZ,invB);
    dg::BZ bZ(gp.R_0,gp.A,psipR,psipZ,psipZZ,psipRZ,invB);
    dg::CurvatureR curvatureR(gp);
    dg::CurvatureZ curvatureZ(gp);
    dg::GradLnB gradLnB(gp);
  
    for (unsigned k=1;k<3;k++) //n iterator
    {
        for (unsigned i=0;i<6;i++) //Nxy iterator
        {
            std::stringstream ss1,ss2;
            ss1 << "dzerr1n" <<k*n<<"Nxy"<<pow(2,i)* Nx<<".txt";
            ss2 << "dzerr2n" <<k*n<<"Nxy"<<pow(2,i)* Nx<<".txt";
            std::string dzerr1fn = ss1.str();
            std::string dzerr2fn = ss2.str();
    //         std::cout << dzerr1fn;
            std::ofstream dzerrfile1((char *) dzerr1fn.c_str());
            std::ofstream dzerrfile2((char *) dzerr2fn.c_str());
            for (unsigned zz=0;zz<5;zz++) //Nz iterator
            {
                std::cout << "n = " << k*n << " Nx = " <<pow(2,i)* Nx << " Ny = " <<pow(2,i)* Ny << " Nz = "<<pow(2,zz)* Nz <<"\n";
                dg::Grid3d<double> g3d( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,k*n,pow(2,i)* Nx,pow(2,i)* Ny, pow(2,zz)*Nz);
                const dg::DVec w3d = dg::create::w3d( g3d);


                std::cout <<"-----(1) test with testfunction" << "\n";
                dg::TestFunction func(psip);
                dg::DeriTestFunction derifunc(psip,psipR,psipZ,ipol,invB);
                std::cout << "Construct parallel  derivative\n";
                dg::Timer t;
                t.tic();
                dg::DZ<dg::DVec> dz( field, g3d); 
                t.toc();
                std::cout << "Creation of parallel Derivative took "<<t.diff()<<"s\n";

        //         std::cout << "Evaluate functions on the grid\n";
                dg::DVec function = dg::evaluate( func, g3d),dzfunc(function);
                dg::DVec diff(g3d.size());

                const dg::DVec solution = dg::evaluate( derifunc, g3d);
                std::cout << "compute parallel z derivative\n";
                t.tic();
                dz( function, dzfunc);
                t.toc();
                std::cout << "Compuation of parallel Derivative took "<<t.diff()<<"s\n";    
                //Compute norm of computed dz
                double normdz = dg::blas2::dot( w3d, dzfunc);
                std::cout << "Norm dz "<<sqrt( normdz)<<"\n";
                //Compute norm of analytical solution
                double normsol = dg::blas2::dot( w3d,solution);
                std::cout << "Norm solution "<<sqrt( normsol)<<"\n";
                dg::blas1::axpby( 1., solution, -1., dzfunc,diff);
                double normdiff = dg::blas2::dot( w3d, diff);
                std::cout << "Norm diff "<<sqrt( normdiff)<<"\n";
                double reldiff=sqrt( dg::blas2::dot( diff, w3d, diff)/normsol );
                std::cout << "Relative Difference Is "<< reldiff<<"\n";

                std::cout <<"-----(2) test with gradlnb" << "\n";    
                //Evaluate analyitcal gradlnb on grid
                dg::DVec gradLnBsolution = dg::evaluate( gradLnB, g3d);
                //compute Ln(B) on grid
                dg::DVec lnBongrid = dg::evaluate( lnB, g3d);
                dg::DVec dzlnBongrid(g3d.size());
                dg::DVec diff2(g3d.size());
                std::cout << "compute parallel z derivative\n";
                t.tic();
                dz(lnBongrid,dzlnBongrid);
                t.toc();
                std::cout << "Compuation of parallel Derivative took "<<t.diff()<<"s\n";    
                //Compute norm of computed dz
                double normdz2 = dg::blas2::dot( w3d, dzlnBongrid);
                std::cout << "Norm dz "<<sqrt( normdz2)<<"\n";
                //Compute norm of analytical gradlnb solution
                double normsol2 = dg::blas2::dot( w3d,gradLnBsolution);
                std::cout << "Norm solution "<<sqrt( normsol2)<<"\n";
                //Compute difference between the solutions
                dg::blas1::axpby( 1., gradLnBsolution , -1., dzlnBongrid,diff2); //diff = gradlnB - dz(ln(B))
                double normdiff2=dg::blas2::dot( w3d, diff2);
                std::cout << "Norm diff "<<sqrt( normdiff2)<<"\n"; //=sqrt(gradlnB - dz(ln(B)))
                double reldiff2 =sqrt( dg::blas2::dot( diff2, w3d, diff2)/normsol2 );
                std::cout << "Relative Difference Is "<<reldiff2 <<"\n";
                dzerrfile1 << pow(2,zz)*Nz <<" " << reldiff << std::endl;
                dzerrfile2 << pow(2,zz)*Nz <<" " << reldiff2 << std::endl;
             }
            dzerrfile1.close();
            dzerrfile2.close();
   
        }

    }

    return 0;
}
