#include <iostream>
#include <iomanip>

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

using Matrix = dg::DMatrix;
using Container = dg::DVec;

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
    dg::Helmholtz<dg::CartesianGrid2d, Matrix, Container> A( alpha, {grid,
            dg::centered});

    {
        t.tic();
        dg::mat::UniversalLanczos< Container > lanczos(x, max_iter);
        //lanczos.set_verbose(true);
        t.toc();
        std::cout << "# Lanczos creation took "<< t.diff()<<"s   \n";

        std::cout << "\nM-Lanczos-T:\n";
        b = dg::evaluate( lhs, grid);
        xexac = dg::evaluate( rhs, grid);
        t.tic();
        unsigned iter = lanczos.solve( x, dg::mat::make_Linear_Te1( 1), A, b,
                w2d, eps, 1., "residual", 1.);
        t.toc();
        dg::blas1::axpby(-1.0, xexac, 1.0, x,error);
        std::cout << "    iter: "<< iter << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x=||b||_M V^T T e_1 and b: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";

        std::cout << "\nM-Lanczos-T-universal:\n";
        b = dg::evaluate( lhs, grid);
        xexac = dg::evaluate( rhs, grid);
        t.tic();
        lanczos.solve( x, dg::mat::make_Linear_Te1( 1), A, b, w2d, eps, 1.,
                "universal", 1., 1);
        t.toc();
        dg::blas1::axpby(-1.0, xexac, 1.0, x,error);
        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x=||b||_M V^T T e_1 and b: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";


        std::cout << "\nM-Lanczos-Tinv:\n";
        b = dg::evaluate( rhs, grid);
        xexac = dg::evaluate( lhs, grid);
        t.tic();
        lanczos.solve( x, dg::mat::make_Linear_Te1( -1), A, b, w2d, eps, 1.,
                "residual");
        t.toc();

        dg::blas1::axpby(-1.0, xexac, 1.0, x,error);
        std::cout << "    iter: "<< lanczos.get_iter() << "\n";
        std::cout << "    time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x=||b||_M V^T T^{-1} e_1 and x: \n";
        std::cout << "    error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";

        std::cout << "\nM-Lanczos-Tinv-universal:\n";
        b = dg::evaluate( rhs, grid);
        xexac = dg::evaluate( lhs, grid);
        t.tic();
        lanczos.solve( x, dg::mat::make_Linear_Te1( -1), A, b, w2d, eps, 1.,
                "universal", 1., 1);
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
        dg::mat::MCG<Container> mcg(x, max_iter);
        //mcg.set_verbose(true);
        t.toc();
        std::cout << "#    M-CG creation took "<< t.diff()<<"s   \n";

        b = dg::evaluate( rhs, grid);
        xexac = dg::evaluate( lhs, grid);
        t.tic();
        auto T = mcg(A, b, w2d, eps);
        auto e1 = mcg.make_e1(), y( e1);
        dg::blas2::symv( dg::mat::invert(T) , e1, y);
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
        std::cout << "    #iter: "<< number << "\n";
        std::cout << "    #time: "<< t.diff()<<"s \n";
        std::cout << "    # Relative error between x=R T e_1 and b: \n";
        std::cout << "    #error: " << sqrt(dg::blas2::dot(w2d, error)/dg::blas2::dot(w2d, xexac)) << " \n";
    }

    return 0;
}
