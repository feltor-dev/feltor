#include <iostream>
#include <iomanip>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif//_OPENMP
#include "algorithm.h"
#include "geometries/geometries.h"


const double lx = 2*M_PI;
const double ly = 2*M_PI;
double lz = 1.;

dg::bc bcx = dg::PER;
dg::bc bcy = dg::PER;
dg::bc bcz = dg::PER;
double left( double x, double y, double z) {return sin(x)*cos(y)*z;}
double right( double x, double y, double z) {return cos(x)*sin(y)*z;}
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y, double z)
{
    return z*z*cos(x)*sin(y)*2*sin(2*x)*cos(2*y)-sin(x)*cos(y)*2*cos(2*x)*sin(2*y);
}

const double R_0 = 1000;
double fct(double x, double y, double z){ return sin(x-R_0)*sin(y);}
double laplace_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y) + 2.*sin(y)*sin(x-R_0);}
double initial( double x, double y, double z) {return sin(0);}

double fct3d(double x, double y, double z){ return sin(x-R_0)*sin(y)*sin(z);}
double laplace_fct3d( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y)*sin(z) + 2.*fct3d(x,y,z) + 1./x/x*fct3d(x,y,z);}

typedef dg::MDMatrix Matrix;
typedef dg::MIDMatrix IMatrix;
typedef dg::MDVec Vector;


/*******************************************************************************
program expects npx, npy, npz, n, Nx, Ny, Nz from std::cin
outputs one line to std::cout
# npx npy npz #procs #threads n Nx Ny Nz t_SCAL t_AXPBY t_POINTWISEDOT t_DOT t_DX_per t_DY_per t_DZ_per t_ARAKAWA #iterations t_1xELLIPTIC_CG_dir_centered t_DS EXBLASCHECK( d and i)
if Nz == 1, DZ and DS are not executed
if std::exception is thrown program writes error to std::cerr and terminates
Run with:
>$ echo npx npy npz n Nx Ny Nz | mpirun -n#procs ./cluster_mpib

 *******************************************************************************/

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert( provided >= MPI_THREAD_FUNNELED && "Threaded MPI lib required!\n");
#else
    MPI_Init(&argc, &argv);
#endif
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    mpi_init3d( bcx, bcy, bcz, n, Nx, Ny, Nz, comm, std::cin, false);
    int rank;
    MPI_Comm_rank( comm, &rank);
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    if(rank==0)
    {
        std::cout<< dims[0] <<" "<<dims[1]<<" "<<dims[2]<<" "<<dims[0]*dims[1]*dims[2];
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads( );
#endif //omp
        std::cout << " "<<num_threads;
        std::cout<<" "<< n <<" "<<Nx<<" "<<Ny<<" "<<Nz;
    }


    dg::CartesianMPIGrid3d grid( 0, lx, 0, ly, 0,lz, n, Nx, Ny, Nz, bcx, bcy, dg::PER, comm);
    periods[0] = false, periods[1] = false;
    MPI_Comm commEll;
    MPI_Cart_create( MPI_COMM_WORLD, 3, dims, periods, true, &commEll);
    if( Nz > 2) lz = 2.*M_PI;
    dg::CylindricalMPIGrid3d gridEll( R_0, R_0+lx, 0., ly, 0.,lz, n, Nx, Ny,Nz, dg::DIR, dg::DIR, dg::PER, commEll);
    dg::Timer t;
    Vector w3d, lhs, rhs, jac, x, y, z;
    try{
        dg::assign( dg::create::weights( grid), w3d);
        dg::assign( dg::evaluate ( left, grid), lhs);
        dg::assign( dg::evaluate ( right,grid), rhs);
        dg::assign( dg::evaluate ( jacobian,grid), jac);
        x = y = z = lhs;
    }
    catch( std::exception& e)
    {
        if(rank==0)std::cout << std::endl;
        if(rank==0)std::cerr << "Caught std::exception: "<<e.what()<<std::endl;
        MPI_Finalize();
        return 0;
    }
    std::cout<< std::setprecision(6);
    unsigned multi=100;

    dg::blas1::scal( x, 3.);//warm up
    //SCAL
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas1::scal( x, 3.);
    t.toc();
    if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    //AXPBY
    dg::blas1::axpby( 3., lhs, 1., jac);//warm up
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas1::axpby( 3., lhs, 1., jac);
    t.toc();
    if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    //PointwiseDot
    dg::blas1::pointwiseDot( 3., lhs,x, 3.,jac, y, 0., z);//warm up
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas1::pointwiseDot( 3., lhs,x, 3.,jac, y, 0., z);
    t.toc();
    if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    //DOT
    double norm=0;
    norm += dg::blas1::dot( lhs, rhs);//warm up
    t.tic();
    for( unsigned i=0; i<multi; i++)
        norm += dg::blas1::dot( lhs, rhs);
    t.toc();
    if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    norm++;//avoid compiler warning
    //Matrix-Vector product
    Matrix dx = dg::create::dx( grid, dg::centered);
    dg::blas2::symv(dx,rhs,jac);//warm up
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas2::symv( dx, rhs, jac);
    t.toc();
    if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    //Matrix-Vector product
    Matrix dy = dg::create::dy( grid, dg::centered);
    dg::blas2::symv(dy,rhs,jac);//warm up
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas2::symv( dy, rhs, jac);
    t.toc();
    if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    if( Nz > 2)
    {
        //Matrix-Vector product
        Matrix dz = dg::create::dz( grid, dg::centered);
        dg::blas2::symv(dz,rhs,jac);//warm up
        t.tic();
        for( unsigned i=0; i<multi; i++)
            dg::blas2::symv( dz, rhs, jac);
        t.toc();
        if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    }
    else
        if(rank==0)std::cout<<" 0.0";
    if(rank==0)std::cout <<std::flush;

    try{

    //The Arakawa scheme
    dg::ArakawaX<dg::CartesianMPIGrid3d, Matrix, Vector> arakawa( grid);
    arakawa( lhs, rhs, jac);//warm up
    t.tic();
    for( unsigned i=0; i<multi; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    if(rank==0)std::cout<<" "<<t.diff()/(double)multi<<std::flush;
    //The Elliptic scheme
    dg::exblas::udouble res;
    if( !(Nz > 2))
    {
        const Vector ellw3d = dg::create::volume(gridEll);
        dg::Elliptic<dg::CylindricalMPIGrid3d, Matrix, Vector> laplace(gridEll, dg::centered);
        const Vector solution = dg::evaluate ( fct, gridEll);
        x = dg::evaluate( initial, gridEll);
        const Vector b = dg::evaluate ( laplace_fct, gridEll);
        dg::PCG< Vector > pcg( x, 1000);
        t.tic();
        unsigned number = pcg.solve(laplace, x, b, 1., ellw3d, 1e-6);
        t.toc();
        if(rank==0)std::cout <<" "<< number << " "<<t.diff()/(double)number<<std::flush;
        dg::blas1::axpby( 1., solution, -1., x);
        //for missing DS
        if(rank==0)std::cout<<" 0.0";
        res.d = dg::blas2::dot( x, ellw3d, x);
    }
    else
    {
        //Elliptic3d
        const Vector ellw3d = dg::create::volume(gridEll);
        dg::Elliptic3d<dg::CylindricalMPIGrid3d, Matrix, Vector> laplace(gridEll, dg::centered);
        const Vector solution = dg::evaluate ( fct3d, gridEll);
        x = dg::evaluate( initial, gridEll);
        const Vector b = dg::evaluate ( laplace_fct3d, gridEll);
        dg::PCG< Vector > pcg( x, multi);
        t.tic();
        unsigned number = pcg.solve(laplace, x, b, 1., ellw3d, 1e-6);
        t.toc();
        if(rank==0)std::cout <<" "<< number << " "<<t.diff()/(double)number<<std::flush;
        dg::blas1::axpby( 1., solution, -1., x);
        res.d = dg::blas2::dot( x, ellw3d, x);
        //Application of ds
        double R0 = 10, I0=20;
        double a = 1.;
        dg::CylindricalMPIGrid3d g3d( R0-a, R0+a, -a, +a, 0, 2.*M_PI, n, Nx
            ,Ny, Nz,dg::DIR, dg::DIR, dg::PER,commEll);
        dg::geo::TokamakMagneticField mag =
            dg::geo::createGuenterField(R0, I0);
        dg::geo::Fieldaligned<dg::aProductMPIGeometry3d, IMatrix, Vector>
            dsFA( mag, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-5, 5, 5);
        dg::geo::DS<dg::aProductMPIGeometry3d, IMatrix, Vector>
            ds ( dsFA);

        ds.centered(x,y);//warm up
        t.tic();
        for( unsigned i=0; i<multi; i++)
            ds.centered(x,y);
        t.toc();
        if(rank==0)std::cout<<" "<<t.diff()/(double)multi;
    }
    if(rank==0)std::cout << " "<<res.d<< " "<<res.i;

    } catch( std::exception& e) {
        if(rank==0)std::cout << std::endl;
        if(rank==0)std::cerr << "Caught std::exception: "<<e.what()<<std::endl;
        MPI_Finalize();
        return 0;
    }

    if(rank==0)std::cout <<std::endl;
    MPI_Finalize();
    return 0;
}
