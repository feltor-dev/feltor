#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>

#include "backend/timer.h"

#include "blas.h"
#include "elliptic.h"
#include "multigrid.h"

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

double initial( double x, double y) {return 0.;}
double amp = 0.9999;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }
//double pol( double x, double y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
//double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}
double der(double x, double y)  { return cos( x)*sin(y);}

//TODO list
//1. Show that the thing reliably converges ( I think we should focus on showing that a single
//FMG sweep reliably produces an acceptable solution even if the discretiation error has not yet been reached)
//2. Can we use this with fixed number of smooting and 1 FMG cycle in simulations?
// (it seems that the number of smoothing steps influences execution time very
// little which means that the CG on the coarse grid dominates time;)
//3. The eps on the lowest grid can be larger than on the fine grids because the discretization error on the coarse grid is h^3 times higher
//4. With forward discretization there seems to be a sweet spot on how many smoothing steps to choose
//5. The relevant errors for us are the gradient in phi errors
//6. The range that the Chebyshev solver smoothes influences the error in the end

int main()
{
    unsigned n = 3, Nx = 32, Ny = 64;
    double eps = 1e-6;
    double jfactor = 1;

    std::cout << "Type n(3) Nx(32) Ny(64)!\n";
    std::cin >> n >> Nx >> Ny;

    std::cout << "Computation on: "<< n <<" x "<< Nx <<" x "<< Ny << std::endl;

    dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    //create functions A(chi) x = b
    dg::DVec x =    dg::evaluate( initial, grid);
    dg::DVec b =    dg::evaluate( rhs, grid);
    const dg::DVec chi =  dg::evaluate( pol, grid);
    const dg::DVec solution = dg::evaluate( sol, grid);

    unsigned stages = 3;
    //std::cout<< "Type number of stages (3) and jfactor (1) !\n";
    //std::cin >> stages >> jfactor;
    //std::cout << stages << " "<<jfactor<<std::endl;
    std::cout << "Type nu1 (20), nu2 (20) gamma (1) \n";
    unsigned nu1, nu2, gamma;
    std::cin >> nu1 >> nu2 >> gamma;
    std::cout << nu1 << " "<<nu2<<" "<<gamma<<std::endl;
    dg::NestedGrids<dg::aGeometry2d, dg::DMatrix, dg::DVec> nested( grid, stages);
    const std::vector<dg::DVec> multi_chi = nested.project( chi);

    std::vector<dg::DVec> multi_x = nested.project( x);
    std::vector<dg::DVec> multi_b = nested.project( b);
    std::vector<dg::EVE<dg::DVec> > multi_eve(stages);
    std::vector<dg::PCG<dg::DVec> > multi_pcg( stages);
    std::vector<dg::ChebyshevIteration<dg::DVec> > multi_cheby( stages);
    std::vector<dg::Elliptic<dg::aGeometry2d, dg::DMatrix, dg::DVec> >
        multi_pol( stages);
    std::vector<std::function<void( const dg::DVec&, dg::DVec&)> >
        multi_inv_pol(stages), multi_inv_cheby(stages), multi_inv_fmg(stages);
    std::vector<double> multi_ev(stages);
    double eps_ev = 1e-10;
    unsigned counter;
    std::cout << "\nPrecision EVE is "<<eps_ev<<"\n";
    for(unsigned u=0; u<stages; u++)
    {
        multi_pol[u].construct( nested.grid(u),
            dg::centered, jfactor);
        multi_pol[u].set_chi( multi_chi[u]);
        multi_pcg[u].construct( multi_x[u], 1000);
        multi_cheby[u].construct( multi_x[u]);

        //estimate EVs
        multi_eve[u].construct( multi_chi[u]);
        counter = multi_eve[u].solve( multi_pol[u], multi_x[u], multi_b[u],
            //multi_pol[u].precond(), multi_pol[u].weights(), multi_ev[u], eps_ev);
            1., multi_pol[u].weights(), multi_ev[u], eps_ev);

        std::cout << "Eigenvalue estimate eve: "<<multi_ev[u]<<"\n";
        std::cout << " with "<<counter<<" iterations\n";
        multi_inv_pol[u] = [&, u, &pcg = multi_pcg[u], &pol = multi_pol[u]](
            const auto& y, auto& x)
            {
                // Watch Back To Basics: Lambdas from Scratch by Arthur O'Dwyer on YT
                // generic lambda (auto makes it a template)
                // init capture (C++14 move objects could be interesting)
                dg::Timer t;
                t.tic();
                int number;
                if ( u == 0)
                    number = pcg.solve( pol, x, y, pol.precond(), pol.weights(), eps,
                        1, 1);
                else
                    number = pcg.solve( pol, x, y, pol.precond(), pol.weights(), eps,
                    1, 10);
                t.toc();
                std::cout << "# Nested iterations stage: " << u << ", iter: " << number << ", took "<<t.diff()<<"s\n";
            };
        auto precond = [nu2, ev = multi_ev[u], &pol = multi_pol[u], &cheby = multi_cheby[u]](
            const auto& y, auto& x)
            {
                //multi_cheby[u].solve( multi_pol[u], x, y, multi_pol[u].precond(),
                cheby.solve( pol, x, y, 1., ev/100., ev*1.1, nu2+1, true);
            };
        multi_inv_cheby[u] = [eps, u, &pcg = multi_pcg[u], &pol = multi_pol[u], p =
            std::move(precond) ]( const auto& y, auto& x)
            {
                dg::Timer t;
                t.tic();
                int number;
                if ( u == 0)
                    number = pcg.solve( pol, x, y, pol.precond(), pol.weights(), eps,
                        1, 1);
                else
                    number = pcg.solve( pol, x, y, p, pol.weights(), eps,
                    1, 10);
                t.toc();
                std::cout << "# Nested iterations stage: " << u << ", iter: " << number << ", took "<<t.diff()<<"s\n";
            };
        if( u == stages-1)
            multi_inv_fmg[u] = [u, eps, &pcg = multi_pcg[u], &pol =
                multi_pol[u] ]( const auto& y, auto& x)
                {
                    dg::Timer t;
                    t.tic();
                    int number = pcg.solve( pol, x, y, pol.precond(), pol.weights(), eps,
                            1, 1);
                    t.toc();
                    std::cout << "# Nested iterations stage: " << u << ", iter: " << number << ", took "<<t.diff()<<"s\n";
                };
        else
            multi_inv_fmg[u] = [nu2, ev = multi_ev[u], u, &cheby =
                multi_cheby[u], &pol = multi_pol[u] ]( const auto& y, auto& x)
                {
                    cheby.solve( pol, x, y, 1., ev/100., ev*1.1, nu2, false);
                };
    }

    std::cout << "\n\n";
    ////////////////////////////////////////////////////
    dg::MultigridCG2d<dg::aGeometry2d, dg::DMatrix, dg::DVec > multigrid(
        grid, stages);
    dg::Timer t;
    std::cout << "MULTIGRID NESTED ITERATIONS SOLVE:\n";
    x = dg::evaluate( initial, grid);
    t.tic();
    //nested_iterations( multi_pol, x, b, multi_inv_pol, nested);
    // same as
    multigrid.solve( multi_pol, x, b, {eps,eps,eps} );
    t.toc();
    double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm);
    std::cout << " Error of nested iterations "<<err<<"\n";
    std::cout << "Took "<<t.diff()<<"s\n\n";
    ////////////////////////////////////////////////////
    std::cout << "MULTIGRID NESTED ITERATIONS WITH CHEBYSHEV SOLVE:\n";
    x = dg::evaluate( initial, grid);
    t.tic();
    dg::nested_iterations( multi_pol, x, b, multi_inv_cheby, nested);
    t.toc();
    norm = dg::blas2::dot( w2d, solution);
    error= solution;
    dg::blas1::axpby( 1.,x,-1., solution, error);
    err = dg::blas2::dot( w2d, error);
    err = sqrt( err/norm);
    std::cout << " Error of nested iterations "<<err<<"\n";
    std::cout << "Took "<<t.diff()<<"s\n\n";
    {
        std::cout << "MULTIGRID PCG SOLVE:\n";
        auto fmg_precond = [&] ( const auto& x, auto& y)
            {
                dg::blas1::copy( 0., y);
                full_multigrid( multi_pol, y,x, multi_inv_fmg, multi_inv_fmg, nested, gamma, 1);
            };
        x = dg::evaluate( initial, grid);
        t.tic();
        multi_pcg[0].solve(multi_pol[0], x, b, fmg_precond, multi_pol[0].weights(), eps);
        t.toc();
        std::cout << "Took "<<t.diff()<<"s\n";
        const double norm = dg::blas2::dot( w2d, solution);
        dg::DVec error( solution);
        dg::blas1::axpby( 1.,x,-1., solution, error);
        double err = dg::blas2::dot( w2d, error);
        err = sqrt( err/norm);
        //std::cout << " At iteration "<<i<<"\n";
        std::cout << " Error of Multigrid iterations "<<err<<"\n\n";
    }
    {
        std::cout << "MULTIGRID FMG SOLVE:\n";
        x = dg::evaluate( initial, grid);
        t.tic();
        //multigrid.fmg_solve(multi_pol, x, b, multi_ev, nu1, nu2, gamma, eps);
        dg::fmg_solve( multi_pol, x, b, multi_inv_fmg, multi_inv_fmg, nested,
                w2d, eps, gamma);
        t.toc();
        std::cout << "Took "<<t.diff()<<"s\n";
        const double norm = dg::blas2::dot( w2d, solution);
        dg::DVec error( solution);
        dg::blas1::axpby( 1.,x,-1., solution, error);
        double err = dg::blas2::dot( w2d, error);
        err = sqrt( err/norm);
        //std::cout << " At iteration "<<i<<"\n";
        std::cout << " Error of Multigrid iterations "<<err<<"\n\n";
    }


    return 0;
}
