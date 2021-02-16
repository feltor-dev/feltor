#include <iostream>
#include <iomanip>

#include "poisson.h"

//const double lx = 2.*M_PI;
//const double ly = 2.*M_PI;



//choose some mean function (attention on lx and ly)
/*
//THESE ARE NOT PERIODIC
double left( double x, double y) { return sin(x/2)*sin(x/2)*exp(x)*sin(y/2.)*sin(y/2.)*log(y+1); }
double right( double x, double y){ return sin(y/2.)*sin(y/2.)*exp(y)*sin(x/2)*sin(x/2)*log(x+1); }
*/
/*
double left( double x, double y) {return sin(x)*exp(x-M_PI)*sin(y);}
double right( double x, double y) {return sin(x)*sin(y);}
double right2( double x, double y) {return exp(y-M_PI);}
double jacobian( double x, double y)
{
    return exp( x-M_PI)*(sin(x)+cos(x))*sin(y) * exp(y-M_PI)*sin(x)*(sin(y) + cos(y)) - sin(x)*exp(x-M_PI)*cos(y) * cos(x)*sin(y)*exp(y-M_PI);
}
*/

double left( double x, double y) {return sin(x)*cos(y);}
double right( double x, double y) { return sin(y)*cos(x); }
const double lx = M_PI;
const double ly = M_PI;
dg::bc bcxlhs = dg::DIR;
dg::bc bcylhs = dg::NEU;
dg::bc bcxrhs = dg::NEU;
dg::bc bcyrhs = dg::DIR;
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y)
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y);
}

int main()
{
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    //![doxygen]
    const dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny);
    const dg::DVec lhs = dg::evaluate( left, grid);
    const dg::DVec rhs = dg::evaluate( right, grid);
    dg::DVec jac(lhs);

    dg::Poisson<dg::aGeometry2d, dg::DMatrix, dg::DVec> poisson( grid, bcxlhs, bcylhs,bcxrhs, bcyrhs );
    poisson( lhs, rhs, jac);
    //![doxygen]

    const dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec eins = dg::evaluate( dg::one, grid);
    const dg::DVec sol = dg::evaluate ( jacobian, grid);
    dg::exblas::udouble res;
    std::cout << std::scientific;
    res.d = dg::blas2::dot( eins, w2d, jac);
    std::cout << "Mean     Jacobian is "<<res.d<<"\t"<<res.i<<"\n";
    res.d = dg::blas2::dot( rhs, w2d, jac);
    std::cout << "Mean rhs*Jacobian is "<<res.d<<"\t"<<res.i<<"\n";
    res.d = dg::blas2::dot( lhs, w2d, jac);
    std::cout << "Mean lhs*Jacobian is "<<res.d<<"\t"<<res.i<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    res.d = sqrt(dg::blas2::dot( w2d, jac));
    std::cout << "Distance to solution "<<res.d<<"\t"<<res.i<<std::endl;
    return 0;
}
