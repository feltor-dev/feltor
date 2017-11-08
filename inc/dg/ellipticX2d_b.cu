#include <iostream>
#include <iomanip>

//#include "backend/xspacelib.cuh"
#include <thrust/device_vector.h>
#include "blas.h"


#include "backend/derivativesX.h"
#include "backend/gridX.h"
#include "backend/evaluationX.cuh"
#include "geometry/base_geometryX.h"
#include "elliptic.h"
#include "cg.h"
#include "backend/timer.cuh"

const dg::bc bcx = dg::DIR;
const dg::bc bcy = dg::NEU;


double initial( double x, double y) {return 0.;}
double sol( double x, double y ) { 
    //if( x < 0)  
    //{
    //    if( y < 0) return sin(x)*sin(y);
    //    else if( 0 <= y && y < 2*M_PI) return sin(x)*cos(y);
    //    else return sin(x)*sin(y - 2*M_PI);
    //}
    return sin(x)*sin(y);
}
double derX( double x, double y) { 
    //if( x < 0)
    //{
    //    if( y < 0) return cos(x)*sin(y);
    //    else if( 0 <= y && y < 2*M_PI) return cos(x)*cos(y);
    //    else return cos(x)*sin(y - 2*M_PI);
    //}
    return cos(x)*sin(y);
}
double derY( double x, double y) { 
    //if( x < 0)
    //{
    //    if( y < 0) return sin(x)*cos(y);
    //    else if( 0 <= y && y < 2*M_PI) return -sin(x)*sin(y);
    //    else return sin(x)*cos(y - 2*M_PI);
    //}
    return sin(x)*cos(y);
}
double lap( double x, double y){ return -2.*sol(x,y); }

double amp = 0.5;
//double pol( double x, double y) {return 1. + amp*sol(x,y); } 
//double rhs( double x, double y) { 
//    return -(1.+amp*sol(x,y))*lap(x,y) - amp*(derX(x,y)*derX(x,y) + derY(x,y)*derY(x,y));}
double pol( double x, double y) {return 1.; } 
double rhs( double x, double y) { return -lap(x,y);} 

typedef dg::DVec Vector;
typedef dg::Composite<dg::EllSparseBlockMatDevice<double> > Matrix;


int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny; 
    double eps;
    std::cout << "Type in n, Nx (1./3.) and Ny (1./6.) and epsilon!\n";
    std::cin >> n >> Nx >> Ny;
    std::cin >> eps;
    std::cout << "Computation on: "<< n <<" x "<<Nx<<" x "<<Ny<<std::endl;
    //std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
    dg::GridX2d grid( -2.*M_PI, M_PI, -M_PI/2., 2.*M_PI+M_PI/2., 1./3., 1./6., n, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    //create functions A(chi) x = b
    dg::DVec x =    dg::evaluate( initial, grid);
    dg::DVec b =    dg::evaluate( rhs, grid);
    dg::DVec chi =  dg::evaluate( pol, grid);
    dg::DVec temp = x;


    std::cout << "Create Polarisation object and set chi!\n";
    t.tic();
    {
    dg::Elliptic<dg::CartesianGridX2d, Matrix, dg::DVec> pol( grid, dg::not_normed, dg::centered);
    pol.set_chi( chi);
    t.toc();
    std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";

    dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny, eps);


    std::cout << eps<<" ";
    t.tic();
    std::cout << " "<< invert( pol, x, b);
    t.toc();
    //std::cout << "Took "<<t.diff()<<"s\n";
    }

    //compute error
    const dg::DVec solution = dg::evaluate( sol , grid);
    const dg::DVec derivati = dg::evaluate( derX, grid);
    dg::DVec error( solution);

    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    //std::cout << "L2 Norm2 of Error is                       " << err << std::endl;
    const double norm = dg::blas2::dot( w2d, solution);
    std::cout << " "<<sqrt( err/norm);
    {
    dg::Elliptic<dg::CartesianGridX2d, Matrix, dg::DVec> pol_forward( grid, dg::not_normed, dg::forward);
    pol_forward.set_chi( chi);
    x = temp;
    dg::Invert<dg::DVec > invert_fw( x, n*n*Nx*Ny, eps);
    std::cout << " "<< invert_fw( pol_forward, x, b);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    std::cout << " "<<sqrt( err/norm);
    }

    {
    dg::Elliptic<dg::CartesianGridX2d, Matrix, dg::DVec> pol_backward( grid, dg::not_normed, dg::backward);
    pol_backward.set_chi( chi);
    x = temp;
    dg::Invert<dg::DVec > invert_bw( x, n*n*Nx*Ny, eps);
    std::cout << " "<< invert_bw( pol_backward, x, b);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    std::cout << " "<<sqrt( err/norm)<<std::endl;
    }


    Matrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    err = dg::blas2::dot( w2d, error);
    //std::cout << "L2 Norm2 of Error in derivative is         " << err << std::endl;
    const double norm_der = dg::blas2::dot( w2d, derivati);
    std::cout << "L2 Norm of relative error in derivative is "<<sqrt( err/norm_der)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2

    return 0;
}

