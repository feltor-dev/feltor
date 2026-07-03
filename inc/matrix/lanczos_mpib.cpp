#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "lanczos.h"
#include "mcg.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -0.5;
const double m = 4.;
const double n = 4.;

double lhs( double x, double y) {return sin(m*x)*sin(n*y);}
double rhs( double x, double y){ return (1.-(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}

using Matrix = dg::MDMatrix;
using Container = dg::MDVec;

int main(int argc, char * argv[])
{
    dg::mpi_init(argc, argv);
    dg::Timer t;
    unsigned n, Nx, Ny;
//     std::cout << "# Type n, Nx and Ny\n";
//     std::cin >> n >> Nx >> Ny;
    MPI_Comm comm;
    dg::mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)
    {
        std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<std::endl;
    }
    unsigned max_iter;
    double eps = 1e-6;
    if(rank==0) std::cout << "# Type in max_iter and eps\n";
    if(rank==0) std::cin >> max_iter>> eps;
    if(rank==0) std::cout <<"# You typed\n"
              <<"max_iter:  "<<max_iter<<"\n"
              <<"eps: "<<eps <<std::endl;
    MPI_Bcast(  &max_iter,1 , dg::getMPIDataType<unsigned>(), 0, comm);
    MPI_Bcast(  &eps,1 , dg::getMPIDataType<double>(), 0, comm);

    dg::RealCartesianMPIGrid2d<double> grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy, comm);

    const Container w2d = dg::create::weights( grid);

    Container x = dg::evaluate( lhs, grid), b(x), error(x), xexac(x);
    dg::Helmholtz<dg::aRealMPIGeometry2d<double>, Matrix, Container> A( alpha, {grid, dg::centered});
    {
        t.tic();
        dg::mat::UniversalLanczos< Container > lanczos(x, max_iter);
        lanczos.set_verbose(true);
        t.toc();
        if(rank==0) std::cout << "# Lanczos creation took "<< t.diff()<<"s   \n";

        if(rank==0) std::cout << "\nM-Lanczos-T:\n";
        b = dg::evaluate( lhs, grid);
        xexac = dg::evaluate( rhs, grid);
        t.tic();
        lanczos.solve( x, dg::mat::make_Linear_Te1( 1), A, b, w2d, eps, 1.,
                "residual", 1.);
        t.toc();
        if(rank==0) std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        if(rank==0) std::cout << "    time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, xexac, 1.0, x,error);
        if(rank==0) std::cout << "    # Relative error between b=||x||_M V^T T e_1 and b: \n";
        double temp = dg::blas2::dot( w2d, error)/dg::blas2::dot( w2d, xexac);
        if(rank==0) std::cout << "    error: " << sqrt(temp) << " \n";

    }
    {
        if(rank==0) std::cout << "\nM-CG: \n";
        t.tic();
        dg::mat::MCG<Container> mcg(x, max_iter);
        t.toc();
        if(rank==0) std::cout << "#    M-CG creation took "<< t.diff()<<"s   \n";
        dg::blas1::scal(x, 0.0); //initialize with zero
        b = dg::evaluate( rhs, grid);
        xexac = dg::evaluate( lhs, grid);
        t.tic();
        auto T = mcg(A, b, w2d, eps);
        auto e1 = mcg.make_e1(), y( e1);
        dg::blas2::symv( dg::mat::invert(T) , e1, y);
        mcg.Ry( A, T, y, x, b);
        t.toc();

        dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
        if(rank==0) std::cout << "    iter: "<< mcg.get_iter() << "\n";
        if(rank==0) std::cout << "    time: "<< t.diff()<<"s \n";
        if(rank==0) std::cout << "    # Relative error between x= R T^{-1} e_1 and x: \n";
        double temp = dg::blas2::dot( w2d, error)/dg::blas2::dot( w2d, xexac);
        if(rank==0) std::cout << "    error: " << sqrt(temp) << " \n";
    }

    MPI_Finalize();
    return 0;
}
