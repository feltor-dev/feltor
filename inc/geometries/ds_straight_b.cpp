#include <iostream>
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#include "ds.h"

double sine(double, double, double z){return sin(z);}
double cosine(double, double, double z){return cos(z);}

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


int main(
#ifdef WITH_MPI
    int argc, char * argv[]
#endif
)
{
#ifdef WITH_MPI
    dg::mpi_init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    DG_RANK0 std::cout << "# Test straight field lines and boundaries in z.\n";

    unsigned n, Nx, Ny, Nz;
#ifdef WITH_MPI
    MPI_Comm comm;
    dg::mpi_init3d( dg::NEU, dg::NEU, dg::PER, n, Nx, Ny, Nz, comm);
#else
    std::cout << "# Type n, Nx, Ny, Nz\n";
    std::cin >> n>> Nx>>Ny>>Nz;
#endif
    DG_RANK0 std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<std::endl;
    dg::x::CartesianGrid3d g3d( -1, 1, -1, 1, 0.1, M_PI+0.1, n, Nx, Ny, Nz, dg::DIR, dg::DIR, dg::NEU
#ifdef WITH_MPI
    , comm
#endif
    );
    dg::x::CartesianGrid2d perp_grid( -1, 1, -1, 1, n, Nx, Ny, dg::DIR, dg::DIR
#ifdef WITH_MPI
    , g3d.get_perp_comm()
#endif
    );
    const dg::x::DVec w3d = dg::create::volume( g3d);
    dg::Timer t;
    t.tic();
    dg::geo::CylindricalVectorLvl1 vec( dg::geo::Constant(0),
            dg::geo::Constant(0), dg::geo::Constant(1), dg::geo::Constant(1),
            dg::geo::Constant(1));

    dg::geo::DS<dg::x::CartesianGrid3d, dg::x::IDMatrix, dg::x::DVec> ds (
            vec, g3d, dg::DIR, dg::DIR, dg::geo::FullLimiter());
    t.toc();
    DG_RANK0 std::cout << "# Creation of parallel Derivative took     "<<t.diff()<<"s\n";

    dg::x::DVec function = dg::evaluate( func, g3d), derivative(function);
    dg::x::DVec constfunc = dg::evaluate( sine, g3d);
    const dg::x::DVec solution = dg::evaluate( deri, g3d);
    const dg::x::DVec constsolution = dg::evaluate( cosine, g3d);
    t.tic();
    ds.set_boundaries( dg::DIR, sin(g3d.z0()),sin(g3d.z1()));
    ds.ds(dg::centered, constfunc, derivative);
    t.toc();
    DG_RANK0 std::cout << "Straight:\n";
    DG_RANK0 std::cout << "# Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., constsolution, -1., derivative);
    double norm = dg::blas2::dot( constsolution, w3d, constsolution);
    double diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    DG_RANK0 std::cout << "    DIR const:   "<< diff<<"\n";
    t.tic();
    ds.set_boundaries( dg::NEU, cos(g3d.z0()),cos(g3d.z1()));
    ds.ds(  dg::centered, constfunc, derivative);
    t.toc();
    DG_RANK0 std::cout << "# Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., constsolution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    DG_RANK0 std::cout << "    NEU const:   "<< diff << "\n";

    t.tic();
    dg::x::DVec left = dg::evaluate( r2, perp_grid), right(left);
    dg::blas1::scal( left, sin(g3d.z0()));
    dg::blas1::scal( right, sin(g3d.z1()));
    ds.set_boundaries( dg::DIR, left,right);
    ds.ds( dg::centered, function, derivative);
    t.toc();
    DG_RANK0 std::cout << "# Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    DG_RANK0 std::cout << "    DIR l/r:     "<< diff << "\n";
    t.tic();
    dg::x::DVec global = dg::evaluate( r2z, g3d);
    ds.set_boundaries( dg::DIR, global, sin(g3d.z0())/(g3d.z0()+g3d.hz()/2.), sin(g3d.z1())/(g3d.z1()-g3d.hz()/2.));
    ds.ds( dg::centered, function, derivative);
    t.toc();
    DG_RANK0 std::cout << "# Application of parallel Derivative took  "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., solution, -1., derivative);
    diff = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm );
    DG_RANK0 std::cout << "    DIR global:  "<< diff <<"\n";

#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
