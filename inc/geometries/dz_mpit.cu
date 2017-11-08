#include <iostream>

#include <cusp/print.h>

#include <mpi.h>
#include "blas.h"
#include "ds.h"
#include "functors.h"

#include "backend/functions.h"
#include "backend/timer.cuh"
#include "backend/mpi_evaluation.h"
#include "backend/mpi_derivatives.h"
#include "backend/mpi_init.h"
#include "geometry.h"

double sine(double x, double y, double z){return sin(z);}
double cosine(double x, double y, double z){return cos(z);}

double func(double x, double y, double z)
{
    double r2 = x*x+y*y;
    return r2*sin(z);
}
double deri(double x, double y, double z)
{
    double r2 = x*x+y*y;
    return r2*cos(z);
}
double r2( double x, double y) {return x*x+y*y;}
double r2z( double x, double y, double z) {return (x*x+y*y)*z;}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::DIR, dg::NEU, n, Nx, Ny, Nz, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::CartesianMPIGrid3d g3d( -1, 1, -1, 1, 0.1, M_PI+0.1, n, Nx, Ny, Nz, dg::DIR, dg::DIR, dg::NEU, comm);
    dg::CartesianMPIGrid2d perp_grid( -1, 1, -1, 1, n, Nx, Ny, dg::DIR, dg::DIR, comm);
    const dg::MDVec w3d = dg::create::volume( g3d);
    dg::Timer t;
    t.tic();
    dg::MDDS::FieldAligned dsFA( dg::DefaultField(), g3d, 1e-10, dg::DefaultLimiter(), dg::NEU);

    dg::MDDS ds ( dsFA, dg::DefaultField(), dg::not_normed, dg::centered);
    t.toc();
    if(rank==0)std::cout << "TEST STRAIGHT FIELD LINES AND BOUNDARIES IN Z\n";
    if(rank==0)std::cout << "Creation of parallel Derivative took     "<<t.diff()<<"s\n";

    dg::MDVec function = dg::evaluate( func, g3d), derivative(function);
    dg::MDVec constfunc = dg::evaluate( sine, g3d);
    const dg::MDVec solution = dg::evaluate( deri, g3d);
    const dg::MDVec constsolution = dg::evaluate( cosine, g3d);
    t.tic();
    ds.set_boundaries( dg::DIR, sin(g3d.z0()),sin(g3d.z1()));
    ds( constfunc, derivative);
    t.toc();
    if(rank==0)std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., constsolution, -1., derivative);
    double norm = dg::blas2::dot( constsolution, w3d, constsolution);
    double diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    if(rank==0)std::cout << "DIR const: Relative Difference Is "<< diff<<"\n";
    t.tic();
    ds.set_boundaries( dg::NEU, cos(g3d.z0()),cos(g3d.z1()));
    ds( constfunc, derivative);
    t.toc();
    if(rank==0)std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., constsolution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    if(rank==0)std::cout << "NEU const: Relative Difference Is "<< diff << "\n";

    t.tic();
    dg::MDVec left = dg::evaluate( r2, perp_grid), right(left);
    dg::blas1::scal( left, sin(g3d.z0()));
    dg::blas1::scal( right, sin(g3d.z1()));
    ds.set_boundaries( dg::DIR, left,right);
    ds( function, derivative);
    t.toc();
    if(rank==0)std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    if(rank==0)std::cout << "DIR l/r: Relative Difference Is "<< diff << "\n";
    t.tic();
    dg::MDVec global = dg::evaluate( r2z, g3d);
    ds.set_boundaries( dg::DIR, global, sin(g3d.z0())/(g3d.z0()+g3d.hz()/2.), sin(g3d.z1())/(g3d.z1()-g3d.hz()/2.));
    ds( function, derivative);
    t.toc();
    if(rank==0)std::cout << "Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    if(rank==0)std::cout << "DIR global: Relative Difference Is "<< diff <<"\n";
    MPI_Finalize();
    
    return 0;
}
