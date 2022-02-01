#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

#include "dg/backend/mpi_init.h"
#include "dg/blas.h"
#include "dg/functors.h"
#include "mpi_evaluation.h"
#include "mpi_derivatives.h"
#include "mpi_weights.h"

double zero( double x, double y) { return 0;}
double sine( double x, double y) { return sin(x)*sin(y);}
double cosx( double x, double y) { return cos(x)*sin(y);}
double cosy( double x, double y) { return cos(y)*sin(x);}
double zero( double x, double y, double z) { return 0;}
double sine( double x, double y, double z) { return sin(x)*sin(y)*sin(z);}
double cosx( double x, double y, double z) { return cos(x)*sin(y)*sin(z);}
double cosy( double x, double y, double z) { return cos(y)*sin(x)*sin(z);}
double cosz( double x, double y, double z) { return cos(z)*sin(x)*sin(y);}


using Matrix = dg::MDMatrix;
using Vector = dg::MDVec;

int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::bc bcx=dg::DIR, bcy=dg::PER, bcz=dg::NEU_DIR;
    if(rank==0)std::cout << "This program tests the creation and application of two-dimensional and three-dimensional derivatives!\n";
    if(rank==0)std::cout << "A TEST is PASSED if the number in the second column shows EXACTLY 0!\n";
    unsigned n = 3, Nx = 24, Ny = 28, Nz = 100;
    if(rank==0)std::cout << "On Grid "<<n<<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<"\n";
    MPI_Comm comm2d;
    mpi_init2d( bcx, bcy, comm2d);
    MPI_Comm comm3d;
    mpi_init3d( bcx, bcy, bcz, comm3d);
    dg::MPIGrid2d g2d( 0, M_PI,0.1, 2*M_PI+0.1, n, Nx, Ny, bcx, bcy, comm2d);
    const Vector w2d = dg::create::weights( g2d);

    Matrix dx2 = dg::create::dx( g2d, dg::forward);
    Matrix dy2 = dg::create::dy( g2d, dg::centered);
    Matrix jx2 = dg::create::jumpX( g2d);
    Matrix jy2 = dg::create::jumpY( g2d);
    Matrix m2[] = {dx2, dy2, jx2, jy2};
    const Vector f2d = dg::evaluate( sine, g2d);
    const Vector dx2d = dg::evaluate( cosx, g2d);
    const Vector dy2d = dg::evaluate( cosy, g2d);
    const Vector null2 = dg::evaluate( zero, g2d);
    Vector sol2[] = {dx2d, dy2d, null2, null2};
    int64_t binary2[] = {4562611930300281864,4553674328256556132,4567083257206218817,4574111364446550002};

    dg::exblas::udouble res;
    if(rank==0)std::cout << "TEST 2D: DX, DY, JX, JY\n";
    for( unsigned i=0; i<4; i++)
    {
        Vector error = sol2[i];
        dg::blas2::symv( -1., m2[i], f2d, 1., error);
        dg::blas1::pointwiseDot( error, error, error);
        double norm = sqrt(dg::blas1::dot( w2d, error)); res.d = norm;
        if(rank==0)std::cout << "Distance to true solution: "<<norm<<"\t"<<res.i-binary2[i]<<"\n";
    }
    dg::MPIGrid3d g3d( 0, M_PI, 0.1, 2*M_PI+0.1, M_PI/2., M_PI, n, Nx, Ny, Nz, bcx, bcy, bcz, comm3d);
    const Vector w3d = dg::create::weights( g3d);
    Matrix dx3 = dg::create::dx( g3d, dg::forward);
    Matrix dy3 = dg::create::dy( g3d, dg::centered);
    Matrix dz3 = dg::create::dz( g3d, dg::backward);
    Matrix jx3 = dg::create::jumpX( g3d);
    Matrix jy3 = dg::create::jumpY( g3d);
    Matrix jz3 = dg::create::jumpZ( g3d);
    Matrix m3[] = {dx3, dy3, dz3, jx3, jy3, jz3};
    const Vector f3d = dg::evaluate( sine, g3d);
    const Vector dx3d = dg::evaluate( cosx, g3d);
    const Vector dy3d = dg::evaluate( cosy, g3d);
    const Vector dz3d = dg::evaluate( cosz, g3d);
    const Vector null3 = dg::evaluate( zero, g3d);
    Vector sol3[] = {dx3d, dy3d, dz3d, null3, null3, null3};
    int64_t binary3[] = {4561946736820639666,4553062895410573431,4594213495911299616,4566393134538626348,4573262464593641240,4594304523193682043};

    if(rank==0)std::cout << "TEST 3D: DX, DY, DZ, JX, JY, JZ\n";
    for( unsigned i=0; i<6; i++)
    {
        Vector error = sol3[i];
        dg::blas2::symv( -1., m3[i], f3d, 1., error);
        double norm = sqrt(dg::blas2::dot( error, w3d, error)); res.d = norm;
        if(rank==0)std::cout << "Distance to true solution: "<<norm<<"\t"<<res.i-binary3[i]<<"\n";
    }
    if(rank==0)std::cout << "TEST if symv captures NaN\n";
    for( unsigned i=0; i<6; i++)
    {
        Vector error = sol3[i];
        error.data()[0] = NAN;
        dg::blas2::symv(  m3[i], f3d, error);
        dg::MPI_Vector<thrust::host_vector<double>> x( error);
        bool hasnan = dg::blas1::reduce( x, false,
                thrust::logical_or<bool>(), dg::ISNFINITE<double>());
        if(rank==0)std::cout << "Symv contains NaN: "<<std::boolalpha<<hasnan<<" (false)\n";
    }
    if(rank==0)std::cout << "\nFINISHED! Continue with arakawa_mpit.cu !\n\n";


    MPI_Finalize();
    return 0;
}
