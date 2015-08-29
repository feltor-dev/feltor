#include <iostream>
#include <iomanip>

#include <mpi.h>
#include "algorithm.h"

#include "backend/timer.cuh"

const double lx = 2*M_PI;
const double ly = 2*M_PI;
const double lz = 1.;

dg::bc bcx = dg::PER;
dg::bc bcy = dg::PER;
dg::bc bcz = dg::PER;
double left( double x, double y, double z) {return sin(x)*cos(y)*z;}
double right( double x, double y, double z) {return cos(x)*sin(y)*z;} 
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y, double z) 
{
    return jacobian(x,y)*z*z;
}
double fct(double x, double y, double z){ return sin(x-R_0)*sin(y);}
double derivative( double x, double y, double z){return cos(x-R_0)*sin(y);}
double laplace_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y) + 2.*sin(y)*sin(x-R_0);}
double initial( double x, double y, double z) {return sin(0);}

typedef dg::MDMatrix Matrix;
typedef dg::MDVec Vector;


//program expects npx, npy, npz, n, Nx, Ny, Nz from std::cin
//outputs one line to std::cout 
// npx, npy, npz, #procs, n, Nx, Ny, Nz, t_AXPBY, t_DOT, t_ARAKAWA, t_1xELLIPTIC

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    int periods[3] = {false,false, false};
    if( bcx == dg::PER) periods[0] = true;
    if( bcy == dg::PER) periods[1] = true;
    if( bcz == dg::PER) periods[2] = true;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    int np[3];
    if( rank == 0)
    {
        std::cin >> np[0] >> np[1]>>np[2];
        std::cout<< np[0] <<" "<<np[1]<<" "<<np[2]<<" "<<size<<" ";
        assert( size == np[0]*np[1]*np[2]);
    }
    MPI_Bcast( np, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
    if( rank == 0)
    {
        std::cin >> n >> Nx >> Ny >> Nz;
        std::cout<< n <<" "<<Nx<<" "<<Ny<<" "<<Nz<<" ";
    }
    MPI_Bcast(  &n,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Nx,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Ny,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Nz,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    ////////////////////////////////////////////////////////////////

    dg::MPI_Grid3d grid( 0, lx, 0, ly, 0,lz, n, Nx, Ny, Nz, bcx, bcy, dg::PER, dg::cartesian, comm);
    dg::Timer t;
    Vector w3d = dg::create::weights( grid);
    Vector v3d = dg::create::inv_weights( grid);
    Vector lhs = dg::evaluate ( left, grid), jac(lhs);
    Vector rhs = dg::evaluate ( right,grid);
    Vector x = dg::evaluate( initial, grid);
    const Vector sol = dg::evaluate( jacobian, grid );
    Vector eins = dg::evaluate( dg::one, grid );
    std::cout<< std::setprecision(6);
    unsigned multi=20;

    //AXPBY
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas1::axpby( 1., lhs, -1., jac);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    //DOT
    t.tic();
    for( unsigned i=0; i<multi; i++)
        norm = dg::blas2::dot( w3d, jac);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    //The Arakawa scheme
    dg::ArakawaX<Matrix, Vector> arakawa( grid);
    t.tic(); 
    for( unsigned i=0; i<multi; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    if(rank==0)std::cout<<t.diff()/(double)multi<<" ";
    //The Elliptic scheme
    periods[0] = false, periods[1] = false;
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
    dg::MPI_Grid3d gridEll( 0, lx, 0, ly, 0,lz, n, Nx, Ny, Nz, dg::DIR, dg::DIR, dg::PER, dg::cartesian, comm);
    dg::Elliptic<Matrix, Vector, Vector> laplace(gridEll, dg::not_normed, dg::centered);
    dg::CG< Vector > pcg( x, n*n*Nx*Ny);
    const Vector solution = dg::evaluate ( fct, gridEll);
    const Vector deriv = dg::evaluate( derivative, gridEll);
    Vector b = dg::evaluate ( laplace_fct, gridEll);
    dg::blas2::symv( w3d, b, b);
    t.tic();
    unsigned number = pcg(laplace, x, b, v3d, eps);
    t.toc();
    if(rank==0)std::cout << number << " "<<t.diff()/(double)number<<" ";


    MPI_Finalize();
    return 0;
}
