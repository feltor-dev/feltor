#include <iostream>
#include <iomanip>

#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "pcg.h"
#include "elliptic.h"

#include "backend/timer.h"

const double lx = M_PI;
const double ly = 2.*M_PI;

double fct(double x, double y){ return sin(y)*sin(x+M_PI/2.);}
double derivative( double x, double y){return cos(x+M_PI/2.)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x+M_PI/2.);}
dg::bc bcx = dg::NEU;
//double fct(double x, double y){ return sin(x);}
//double derivative( double x, double y){return cos(x);}
//double laplace_fct( double x, double y) { return sin(x);}
//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;
double initial( double x, double y) {return sin(0);}


int main( int argc, char* argv[])
{
    dg::Timer t;
    unsigned n, Nx, Ny;
    double eps=1e-6;
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    dg::mpi_init2d( bcx, dg::PER, n, Nx, Ny, comm);
    if(rank==0)std::cout << "Type epsilon! \n";
    if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , MPI_DOUBLE, 0, comm);
    dg::x::Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER,comm );
    //dg::x::RealGrid2d<float> gridf( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER, comm);
#else
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Type epsilon! \n";
    std::cin >> eps;
    dg::x::Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    //dg::x::RealGrid2d<float> gridf( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
#endif
    DG_RANK0 std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;

    const dg::x::DVec w2d = dg::create::weights( grid);
    const dg::x::DVec v2d = dg::create::inv_weights( grid);
    DG_RANK0 std::cout<<"Evaluate initial condition...\n";
    dg::x::DVec x = dg::evaluate( initial, grid);

    std::cout << "Create Laplacian...\n";
    t.tic();
    dg::x::DMatrix DX = dg::create::dx( grid);
    dg::Elliptic<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec> lap(grid,
        dg::forward );
    //dg::Elliptic<dg::x::aRealGeometry2d<float>, dg::x::fDMatrix,
    //    dg::x::fDVec> flap(gridf, dg::forward ); // never used?
    t.toc();
    DG_RANK0 std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::PCG pcg( x, n*n*Nx*Ny);

    DG_RANK0 std::cout<<"Expand right hand side\n";
    const dg::x::DVec solution = dg::evaluate ( fct, grid);
    const dg::x::DVec deriv = dg::evaluate( derivative, grid);
    dg::x::DVec b = dg::evaluate ( laplace_fct, grid);
    //////////////////////////////////////////////////////////////////////

    x = dg::evaluate( initial, grid);
    t.tic();
    int number = pcg.solve( lap, x, b, 1., w2d, eps);
    t.toc();
    DG_RANK0
    {
        std::cout << "# of pcg itersations   "<<number<<std::endl;
        std::cout << "... for a precision of "<< eps<<std::endl;
        std::cout << "...               took "<< t.diff()<<"s\n";
    }

    dg::x::DVec error( solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w2d, error);
    double norm = dg::blas2::dot( w2d, solution);
    DG_RANK0 std::cout << "L2 Norm of relative error is: " <<sqrt( normerr/norm)<<std::endl;
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w2d, error);
    norm = dg::blas2::dot( w2d, deriv);
    DG_RANK0 std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    //both functiona and derivative converge with order P

#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
