#define DG_DEBUG
#include <iostream>
#include <iomanip>

#include "lanczos.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -1;
const double m = 1.;
const double n = 1.;

double lhs( double x, double y) {return sin(m*x)*sin(n*y);}
double rhs( double x, double y){ return (1.-(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}

using Matrix = dg::DMatrix;
using Container = dg::DVec;
using HDiaMatrix = cusp::dia_matrix<int, double, cusp::host_memory>;
using HCooMatrix = cusp::coo_matrix<int, double, cusp::host_memory>;

int main(int argc, char * argv[])
{
    dg::Timer t;
    unsigned n, Nx, Ny;
    std::cout << "# Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<std::endl;
    unsigned max_iter;
    std::cout << "# Type in max_iter and eps\n";
    double eps = 1e-6;
    std::cin >> max_iter>> eps;
    std::cout <<"# You typed\n"
              <<"max_iter:  "<<max_iter<<"\n"
              <<"eps: "<<eps <<std::endl;
    dg::CartesianGrid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy);
    const Container w2d = dg::create::weights( grid);
    Container x = dg::evaluate( lhs, grid), b(x), error(x),  xexac(x);
    Container bexac = dg::evaluate( rhs, grid);
    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( grid, alpha, dg::centered);

    {
        t.tic();
        dg::Lanczos< Container > lanczos(x, max_iter);
        lanczos.set_verbose(true);
        t.toc();
        std::cout << "# Lanczos creation took "<< t.diff()<<"s   \n";

        HDiaMatrix T;
        std::cout << "Lanczos:\n";

        t.tic();
        T = lanczos( A, x, b, 1., eps, true);
        t.toc();

        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
        std::cout << "    # Relative error between b=||x||_2 V^T T e_1 and b: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";

        std::cout << "\nM-Lanczos:\n";
        x = dg::evaluate( lhs, grid);
        t.tic();
        T = lanczos(A, x, b, w2d, eps, true);
        t.toc();
        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
        std::cout << "    # Relative error between b=||x||_M V^T T e_1 and b: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";

    }
    {
        std::cout << "\nM-CG: \n";
        t.tic();
        dg::MCG<Container> mcg(x, max_iter);
        t.toc();
        std::cout << "#    M-CG creation took "<< t.diff()<<"s   \n";
//         dg::blas1::scal(x, 0.0); //initialize with zero
        dg::blas1::scal(x, 0.0); //initialize with zero
        //dg::blas1::copy(1000., x); //initialize not with zero

        t.tic();
        HDiaMatrix T = mcg(A, x, b, w2d, eps, 1., true);
        t.toc();


        dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
        std::cout << "    iter: "<< mcg.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x= R T^{-1} e_1 and x: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";
    }

    return 0;
}
