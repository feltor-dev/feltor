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
    Container x = dg::evaluate( lhs, grid), error(x);
    Container b = dg::evaluate( rhs, grid), xexac(x);
    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( grid, alpha,
            dg::centered);

    {
        t.tic();
        dg::Lanczos< Container > lanczos(x, max_iter);
        //lanczos.set_verbose(true);
        t.toc();
        std::cout << "# Lanczos creation took "<< t.diff()<<"s   \n";

        HDiaMatrix T;
        // This of course does not work because A is not self-adjoint in 1
        //std::cout << "Lanczos:\n";

        //t.tic();
        //T = lanczos( A, b, 1., eps);
        //t.toc();

        //std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        //std::cout << "    time: "<< t.diff()<<"s \n";
        //dg::blas1::axpby(-1.0, bexac, 1.0, b,error);
        //std::cout << "    # Relative error between b=||x||_2 V^T T e_1 and b: \n";
        //std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, bexac)) << " \n";

        std::cout << "\nM-Lanczos-T:\n";
        b = dg::evaluate( lhs, grid);
        xexac = dg::evaluate( rhs, grid);
        t.tic();
        T = lanczos(A, b, w2d, eps);
        //Check if a vector contains Inf or NaN
        auto e1 = lanczos.make_e1(), y( e1);
        dg::blas2::symv( T , e1, y);
        lanczos.normMbVy( A, T, y, x, b, lanczos.get_bnorm());
        t.toc();
        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        dg::blas1::axpby(-1.0, xexac, 1.0, x,error);
        std::cout << "    # Relative error between x=||b||_M V^T T e_1 and b: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";

        std::cout << "\nM-Lanczos-Tinv:\n";
        b = dg::evaluate( rhs, grid);
        xexac = dg::evaluate( lhs, grid);
        t.tic();
        T = lanczos(A, b, w2d, eps);
        e1 = lanczos.make_e1(), y = e1;
        dg::blas2::symv( dg::invert( T) , e1, y);
        lanczos.normMbVy( A, T, y, x, b, lanczos.get_bnorm());
        t.toc();

        dg::blas1::axpby(-1.0, xexac, 1.0, x,error);
        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x=||b||_M V^T T^{-1} e_1 and x: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";

    }
    {
        std::cout << "\nM-CG-Tinv: \n";
        t.tic();
        dg::MCG<Container> mcg(x, max_iter);
        //mcg.set_verbose(true);
        t.toc();
        std::cout << "#    M-CG creation took "<< t.diff()<<"s   \n";

        b = dg::evaluate( rhs, grid);
        xexac = dg::evaluate( lhs, grid);
        t.tic();
        HDiaMatrix T = mcg(A, b, w2d, eps);
        auto e1 = mcg.make_e1(), y( e1);
        dg::blas2::symv( dg::invert(T) , e1, y);
        mcg.Ry( A, T, y, x, b);
        t.toc();

        dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
        std::cout << "    iter: "<< mcg.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x= R T^{-1} e_1 and x: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";
        std::cout << "\nM-CG-T: \n";
        b = dg::evaluate( lhs, grid);
        xexac = dg::evaluate( rhs, grid);
        t.tic();
        T = mcg(A, b, w2d, eps);
        e1 = mcg.make_e1(), y = e1;
        dg::blas2::symv( T, e1, y);
        mcg.Ry( A, T, y, x, b);
        t.toc();

        dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
        std::cout << "    iter: "<< mcg.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x=R T e_1 and b: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";

        std::cout << "\n#Test solution of normal PCG\n";
        dg::PCG<Container> pcg( x, max_iter);
        b = dg::evaluate( rhs, grid);
        xexac = dg::evaluate( lhs, grid);
        dg::blas1::copy( 0., x);
        t.tic();
        unsigned number = pcg.solve(A, x, b, 1., w2d, eps);
        t.toc();

        dg::blas1::axpby(-1.0, xexac, 1.0, x, error);
        std::cout << "    #iter: "<< mcg.get_iter() << "\n";
        std::cout << "    #time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x=R T e_1 and b: \n";
        std::cout << "    #error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";
    }

    return 0;
}
