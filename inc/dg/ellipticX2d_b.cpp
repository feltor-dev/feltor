#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include "backend/timer.h"
#include "topology/derivativesX.h"
#include "topology/gridX.h"
#include "topology/evaluationX.h"
#include "topology/base_geometryX.h"
#include "blas.h"
#include "elliptic.h"
#include "pcg.h"

const dg::bc bcx = dg::DIR;
const dg::bc bcy = dg::NEU;


double initial( double, double) {return 0.;}
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
double pol( double, double) {return 1.; }
double rhs( double x, double y) { return -lap(x,y);}

typedef dg::DVec Vector;
typedef dg::Composite<dg::EllSparseBlockMat<double, thrust::device_vector> > Matrix;


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
    dg::PCG<dg::DVec > pcg( x, n*n*Nx*Ny);


    std::cout << "Create Polarisation object and set chi!\n";
    t.tic();
    {
    dg::Elliptic<dg::CartesianGridX2d, Matrix, dg::DVec> pol( grid, dg::centered);
    pol.set_chi( chi);
    t.toc();
    std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";

    std::cout << eps<<" ";
    t.tic();
    std::cout << " "<< pcg.solve( pol, x, b, pol.precond(), pol.weights(), eps);
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
    dg::Elliptic<dg::CartesianGridX2d, Matrix, dg::DVec> pol_forward( grid, dg::forward);
    pol_forward.set_chi( chi);
    x = temp;
    std::cout << " "<< pcg.solve( pol_forward, x, b, pol_forward.precond(), pol_forward.weights());
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    std::cout << " "<<sqrt( err/norm);
    }

    {
    dg::Elliptic<dg::CartesianGridX2d, Matrix, dg::DVec> pol_backward( grid, dg::backward);
    pol_backward.set_chi( chi);
    x = temp;
    std::cout << " "<< pcg.solve( pol_backward, x, b, pol_backward.precond(), pol_backward.weights());
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

