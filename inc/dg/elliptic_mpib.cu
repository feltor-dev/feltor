#include <iostream>
#include <iomanip>

#include <mpi.h>

#include "cg.h"
#include "elliptic.h"

#include "backend/timer.cuh"
#include "backend/mpi_init.h"
#include "geometry/split_and_join.h"


const double R_0 = 1000;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 2.*M_PI;
double fct( double x, double y, double z){ return sin(x-R_0)*sin(y);}
double derivative( double x, double y, double z){return cos(x-R_0)*sin(y);}
double laplace_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y) + 2.*sin(y)*sin(x-R_0);}
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
double initial( double x, double y, double z) {return sin(0);}


int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    dg::mpi_init3d( bcx, bcy, dg::PER, n, Nx, Ny, Nz, comm);

    dg::CylindricalMPIGrid3d grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, bcy, dg::PER, comm);
    const dg::MDVec w3d = dg::create::volume( grid);
    const dg::MDVec v3d = dg::create::inv_volume( grid);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    double eps=1e-6;
    if(rank==0)std::cout << "Type epsilon! \n";
    if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , MPI_DOUBLE, 0, comm);
    /////////////////////////////////////////////////////////////////
    if(rank==0)std::cout<<"TEST CYLINDRIAL LAPLACIAN!\n";
    dg::Timer t;
    dg::MDVec x = dg::evaluate( initial, grid);

    if(rank==0)std::cout << "Create Laplacian\n";
    t.tic();
    dg::Elliptic<dg::CylindricalMPIGrid3d, dg::MDMatrix, dg::MDVec> laplace(grid, dg::not_normed, dg::centered);
    dg::MDMatrix DX = dg::create::dx( grid);
    t.toc();
    if(rank==0)std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::MDVec > pcg( x, n*n*Nx*Ny);

    if(rank==0)std::cout<<"Expand right hand side\n";
    const dg::MDVec solution = dg::evaluate ( fct, grid);
    const dg::MDVec deriv = dg::evaluate( derivative, grid);
    dg::MDVec b = dg::evaluate ( laplace_fct, grid);
    //compute W b
    dg::blas2::symv( w3d, b, b);

    if(rank==0)std::cout << "For a precision of "<< eps<<" ..."<<std::endl;
    t.tic();
    unsigned num = pcg( laplace,x,b,v3d,eps);
    if(rank==0)std::cout << "Number of pcg iterations "<< num<<std::endl;
    t.toc();
    if(rank==0)std::cout << "... took                 "<< t.diff()<<"s\n";
    dg::MDVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w3d, error);
    double norm = dg::blas2::dot( w3d, solution);
    if(rank==0)std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, deriv);
    if(rank==0)std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;

    if(rank==0)std::cout << "TEST SPLIT SOLUTION\n";
    x = dg::evaluate( initial, grid);
    b = dg::evaluate ( laplace_fct, grid);
    //create grid and perp and parallel volume
    dg::Handle<dg::aMPIGeometry2d> grid_perp = grid.perp_grid();
    dg::MDVec v2d = dg::create::inv_volume( grid_perp.get());
    dg::MDVec w2d = dg::create::volume( grid_perp.get());
    dg::SparseElement<dg::MDVec> g_parallel = dg::tensor::volume( grid.metric().parallel());
    dg::MDVec chi = dg::evaluate( dg::one, grid);
    dg::tensor::pointwiseDot( chi, g_parallel, chi);
    //create split Laplacian
    std::vector< dg::Elliptic<dg::aMPIGeometry2d, dg::MDMatrix, dg::MDVec> > laplace_split(
            grid.local().Nz(), dg::Elliptic<dg::aMPIGeometry2d, dg::MDMatrix, dg::MDVec>(grid_perp.get(), dg::not_normed, dg::centered));
    // create split  vectors and solve
    std::vector<dg::MDVec> b_split, x_split, chi_split;
    pcg.construct( w2d, w2d.size());
    std::vector<unsigned>  number(grid.local().Nz());
    t.tic();
    dg::tensor::pointwiseDot( b, g_parallel, b);
    dg::split( b, b_split, grid);
    dg::split( chi, chi_split, grid);
    dg::split( x, x_split, grid);
    for( unsigned i=0; i<grid.local().Nz(); i++)
    {
        laplace_split[i].set_chi( chi_split[i]);
        dg::blas1::pointwiseDot( b_split[i], w2d, b_split[i]);
        number[i] = pcg( laplace_split[i], x_split[i], b_split[i], v2d, eps);
    }
    dg::join( x_split, x, grid);
    t.toc();
    if(rank==0)std::cout << "Number of iterations in split     "<< number[0]<<"\n";
    if(rank==0)std::cout << "Split solution on the device took "<< t.diff()<<"s\n";
    dg::blas1::axpby( 1., x,-1., solution, error);
    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, solution);
    if(rank==0)std::cout << "L2 Norm of relative error is:     " <<sqrt( normerr/norm)<<std::endl;
    //both function and derivative converge with order P

    MPI_Finalize();
    return 0;
}
