#include <iostream>
#include <iomanip>
#include <vector>

// #include "typedefs.cuh"
#include "xspacelib.cuh"
#include "geometry.cuh"

#include <fstream>
#include <sstream>
#include "file/read_input.h"

#include <cusp/print.h>

#include "dz.cuh"
#include "rk.cuh"
#include "functions.h"
#include "interpolation.cuh"
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
    std::cout << "Type n, Nx, Ny, Nz\n";
    std::cin >> n>> Nx>>Ny>>Nz;
//     n=1;Nx=100;Ny=100;Nz=15;
//     Nxh=Nx/2;
    
//     Nyh=Ny/2;
//     Rmin=R_0-1;
//     Zmin=-Nyh;
//     Rmax=R_0+1; 
//     Zmax=Nyh;
    std::cout << "The grid parameters" <<"\n";
    std::cout  << Rmin<<"  " << Rmax <<"  " << Zmin <<"  " <<Zmax <<"\n";
//     std::cin >> n>> Nx>>Ny>>Nz;
    dg::Grid3d<double> g3d( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, n, Nx, Ny, Nz);
    const dg::DVec w3d = dg::create::w3d( g3d);
    dg::Field field(gp);
    dg::Psip psip(gp.R_0,gp.A,gp.c);
    dg::PsipR psipR(gp.R_0,gp.A,gp.c);
    dg::PsipRR psipRR(gp.R_0,gp.A,gp.c);  
    dg::PsipZ psipZ(gp.R_0,gp.A,gp.c);  
    dg::PsipZZ psipZZ(gp.R_0,gp.A,gp.c);   
    dg::PsipRZ psipRZ(gp.R_0,gp.A,gp.c);  
    dg::Ipol ipol(gp.R_0,gp.A,psip);
    dg::InvB invB(gp.R_0,ipol,psipR,psipZ);
    dg::BR bR(gp.R_0,gp.A,psipR,psipRR,psipZ,psipRZ,invB);
    dg::BZ bZ(gp.R_0,gp.A,psipR,psipZ,psipZZ,psipRZ,invB);
    dg::CurvatureR curvatureR(invB,bZ);
    dg::CurvatureZ curvatureZ(invB,bR);
    dg::GradLnB gradLnB(invB, psipR,psipZ,bR,bZ);

    dg::TestFunction func(psip);
    dg::DeriTestFunction deri(psip,psipR,psipZ,ipol,invB);
    std::cout << "Construct z derivative\n";
    dg::DZ<dg::DVec> dz( field, g3d); 
    std::cout << "Evaluate functions on the grid\n";
    dg::DVec function = dg::evaluate( func, g3d),derivative(function);
   
    const dg::DVec solution = dg::evaluate( deri, g3d);

    dz( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( w3d, solution);
    std::cout << "Norm Solution "<<sqrt( norm)<<"\n";
//     double norm2 = dg::blas2::dot( w3d, solution);
    std::cout << "Relative Difference Is "<< sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm )<<"\n";    
    return 0;
}
