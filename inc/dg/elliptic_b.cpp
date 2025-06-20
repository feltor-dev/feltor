#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "backend/timer.h"
#include "topology/evaluation.h"
#include "topology/split_and_join.h"

#include "pcg.h"
#include "elliptic.h"


const double R_0 = 10;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 2.*M_PI;
double fct(double x, double y, double z){ return sin(x-R_0)*sin(y)*sin(z);}
double fctX( double x, double y, double z){return cos(x-R_0)*sin(y)*sin(z);}
double fctY(double x, double y, double z){ return sin(x-R_0)*cos(y)*sin(z);}
double fctZ(double x, double y, double z){ return sin(x-R_0)*sin(y)*cos(z);}
// Cartesian Laplace
double fctC(double x, double y, double z){ return sin(x)*sin(y)*sin(z);}
double laplace_fctC( double x, double y, double z) { return 2*sin(y)*sin(x)*sin(z);}
// Cylindrical Laplace
double laplace2d_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y)*sin(z) + 2.*fct(x,y,z);}
double laplace3d_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y)*sin(z) + 2.*fct(x,y,z) + 1./x/x*fct(x,y,z);}
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
dg::bc bcz = dg::PER;
double initial( double, double, double) {return sin(0);}
double variation3d( double x, double y, double z) {
    return (fctX(x,y,z)*fctX(x,y,z)
        + fctY(x,y,z)*fctY(x,y,z)
        + fctZ(x,y,z)*fctZ(x,y,z)/x/x)*fct(x,y,z)*fct(x,y,z);
}


int main(
#ifdef WITH_MPI
    int argc, char* argv[]
#endif
)
{
    unsigned n, Nx, Ny, Nz;
    double eps=1e-6;
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    dg::mpi_init3d( bcx, bcy, bcz, n, Nx, Ny, Nz, comm);
    if(rank==0)std::cout << "Type epsilon! \n";
    if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , MPI_DOUBLE, 0, comm);
#else
    std::cout << "Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    std::cout << "Type epsilon! \n";
    std::cin >> eps;
    //bool jump_weight;
    //std::cout << "Jump weighting on or off? Type 1 for true or 0 for false (default): \n";
    //std::cin >> jump_weight;
#endif

    dg::Timer t;
    { // Cartesian3d
    DG_RANK0 std::cout << "Test Cartesian Laplacian\n";
    dg::x::CartesianGrid3d g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, bcx, bcy, bcz
#ifdef WITH_MPI
    , comm
#endif
    );
    const dg::x::DVec w3d = dg::create::weights( g3d);
    dg::x::DVec x3 = dg::evaluate( initial, g3d);
    dg::x::DVec b3 = dg::evaluate ( laplace_fctC, g3d);
    dg::Elliptic<dg::x::CartesianGrid3d, dg::x::DMatrix, dg::x::DVec> lap(g3d, dg::forward );
    dg::PCG pcg( x3, g3d.size());
    t.tic();
    unsigned num = pcg.solve( lap, x3, b3, 1., w3d, eps, sqrt(lz));
    t.toc();
    DG_RANK0 std::cout << "Number of pcg iterations "<< num<<std::endl;
    DG_RANK0 std::cout << "... for a precision of   "<< eps<<std::endl;
    DG_RANK0 std::cout << "... on the device took   "<< t.diff()<<"s\n";
    t.tic();
    //compute error
    const dg::x::DVec solution3 = dg::evaluate ( fctC, g3d);
    dg::x::DVec error3( solution3);
    dg::blas1::axpby( 1.,x3,-1.,error3);

    double eps3 = dg::blas2::dot(w3d , error3);
    double norm3 = dg::blas2::dot(w3d , solution3);
    DG_RANK0 std::cout << "L2 Norm of relative error is:  " <<sqrt( eps3/norm3)<<std::endl;
    }
    { // Cylindrical3d
    DG_RANK0 std::cout << "Test Cylindrical Laplacian\n";
    //! [invert]
    dg::x::CylindricalGrid3d grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, bcy, bcz
#ifdef WITH_MPI
    , comm
#endif
    );
    const dg::x::DVec w3d = dg::create::volume( grid);
    dg::x::DVec x = dg::evaluate( initial, grid);

    dg::Elliptic3d<dg::x::aGeometry3d, dg::x::DMatrix, dg::x::DVec>
        laplace( grid, dg::centered);

    dg::PCG pcg( x, n*n*Nx*Ny*Nz);

    const dg::x::DVec solution = dg::evaluate ( fct, grid);
    const dg::x::DVec b = dg::evaluate ( laplace3d_fct, grid);

    DG_RANK0 std::cout << "For a precision of "<< eps<<" ..."<<std::endl;
    t.tic();
    unsigned num = pcg.solve( laplace, x, b, 1., w3d, eps);
    t.toc();
    DG_RANK0 std::cout << "Number of pcg iterations "<<num<<std::endl;
    DG_RANK0 std::cout << "... on the device took   "<< t.diff()<<"s\n";
    //! [invert]
    dg::x::DVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w3d, error);
    double norm = dg::blas2::dot( w3d, solution);
    dg::exblas::udouble res;
    norm = sqrt(normerr/norm); res.d = norm;
    DG_RANK0 std::cout << "L2 Norm of relative error is:               " <<res.d<<"\t"<<res.i<<std::endl;
    const dg::x::DVec deriv = dg::evaluate( fctX, grid);
    dg::x::DMatrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, deriv);
    DG_RANK0 std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    DG_RANK0 std::cout << "Compute variation in Elliptic               ";
    const dg::x::DVec variatio = dg::evaluate ( variation3d, grid);
    laplace.variation( solution, x, error);
    dg::blas1::axpby( 1., variatio, -1., error);
    norm = dg::blas2::dot( w3d, variatio);
    normerr = dg::blas2::dot( w3d, error);
    DG_RANK0 std::cout <<sqrt( normerr/norm) << "\n";
    }

    { // Split solution
    DG_RANK0 std::cout << "Test split solution\n";
    dg::x::CylindricalGrid3d grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, bcy, bcz
#ifdef WITH_MPI
    , comm
#endif
    );
    const dg::x::DVec w3d = dg::create::volume( grid);
    dg::x::DVec x = dg::evaluate( initial, grid);
    dg::x::DVec b = dg::evaluate ( laplace2d_fct, grid);
    //create grid and perp and parallel volume
    dg::ClonePtr<dg::x::aGeometry2d> grid_perp = grid.perp_grid();
    const dg::x::DVec w2d = dg::create::volume( *grid_perp);
    dg::x::DVec g_parallel = grid.metric().value(2,2);
    dg::blas1::transform( g_parallel, g_parallel, dg::SQRT<>());
    dg::x::DVec chi = dg::evaluate( dg::one, grid);
    dg::blas1::pointwiseDivide( chi, g_parallel, chi);
    //create split Laplacian
    std::vector< dg::Elliptic<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec> > laplace_split(
#ifdef WITH_MPI
            grid.local().Nz(),
#else
            grid.Nz(),
#endif
            {*grid_perp, dg::centered});
    // create split  vectors and solve
    dg::PCG pcg( w2d, grid_perp->size());
    std::vector<unsigned>  number(grid.Nz());
    t.tic();
    dg::blas1::pointwiseDivide( b, g_parallel, b);
    auto b_split   = dg::split( b, grid);
    auto chi_split = dg::split( chi, grid);
    auto x_split   = dg::split( x, grid);
    for( unsigned i=0; i<laplace_split.size(); i++)
    {
        laplace_split[i].set_chi( chi_split[i]);
        number[i] = pcg.solve( laplace_split[i], x_split[i], b_split[i], 1., w2d, eps);
    }
    t.toc();
    DG_RANK0 std::cout << "Number of iterations in split     "<< number[0]<<"\n";
    DG_RANK0 std::cout << "Split solution on the device took "<< t.diff()<<"s\n";
    const dg::x::DVec solution = dg::evaluate ( fct, grid);
    dg::blas1::axpby( 1., x,-1., solution, x);
    double normerr = dg::blas2::dot( w3d, x);
    double norm = dg::blas2::dot( w3d, solution);
    DG_RANK0 std::cout << "L2 Norm of relative error is:     " <<sqrt( normerr/norm)<<std::endl;
    }

    //both function and derivative converge with order P
#ifdef WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}
