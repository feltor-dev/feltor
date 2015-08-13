#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "backend/xspacelib.cuh"
#include "backend/timer.cuh"
#include "backend/mpi_init.h"

#include "elliptic.h"
#include "cg.h"


//NOTE: IF DEVICE=CPU THEN THE POLARISATION ASSEMBLY IS NOT PARALLEL AS IT IS NOW 

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI;
const double ly = M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
//const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double amp = 1;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }
//double pol( double x, double y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
//double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}
double der(double x, double y)  { return cos( x)*sin(y);}


int main(int argc, char* argv[] )
{
    /////////////////////MPI INIT//////////////////////////
    MPI_Init( &argc, &argv);
    unsigned n, Nx, Ny; 
    MPI_Comm comm;
    mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::Timer t;
    double eps = 1e-6;
    //if(rank==0)std::cout << "Type epsilon! \n";
    //if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , MPI_DOUBLE, 0, comm);
    //////////////////////begin program///////////////////////
    //create functions A(chi) x = b
    dg::MPI_Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy, comm);
    const dg::MPrecon w2d = dg::create::weights( grid);
    const dg::MPrecon v2d = dg::create::inv_weights( grid);
    dg::MVec x =    dg::evaluate( initial, grid);
    dg::MVec b =    dg::evaluate( rhs, grid);
    dg::MVec chi =  dg::evaluate( pol, grid);


    //if(rank==0)std::cout << "Create Polarisation object and set chi!\n";
    t.tic();
    dg::Elliptic<dg::MMatrix, dg::MVec, dg::MPrecon> pol( grid, dg::not_normed, dg::centered);
    pol.set_chi( chi);
    t.toc();
    //if(rank==0)std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";

    dg::Invert<dg::MVec > invert( x, n*n*Nx*Ny, eps);
    t.tic();
    unsigned number = invert( pol, x, b);
    t.toc();
    //if(rank==0)std::cout << "Number of pcg iterations "<< number<<std::endl;
    //if(rank==0)std::cout << "For a precision of "<< eps<<std::endl;
    if(rank==0)std::cout << " took "<<t.diff()<<"s\n";

    //compute error
    const dg::MVec solution = dg::evaluate( sol, grid);
    const dg::MVec derivati = dg::evaluate( der, grid);
    dg::MVec error( solution);
    dg::blas1::axpby( 1.,x,-1., error);

    double err = dg::blas2::dot( w2d, error);
    //if(rank==0)std::cout << "L2 Norm2 of Error is " << err << std::endl;
    double norm = dg::blas2::dot( w2d, solution);
    //if(rank==0)std::cout << "L2 Norm of relative error is               "<<sqrt( err/norm)<<std::endl;
    dg::MMatrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    err = dg::blas2::dot( w2d, error);
    //if(rank==0)std::cout << "L2 Norm2 of Error in derivative is " << err << std::endl;
    norm = dg::blas2::dot( w2d, derivati);
    //if(rank==0)std::cout << "L2 Norm of relative error in derivative is "<<sqrt( err/norm)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2

    MPI_Finalize();
    return 0;
}

