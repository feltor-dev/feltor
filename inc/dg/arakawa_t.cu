#include <iostream>
#include <iomanip>

#include "arakawa.h"

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

//![function]
const double lx = 2*M_PI;
const double ly = 2*M_PI;
dg::bc bcx = dg::PER; 
dg::bc bcy = dg::PER;
double left( double x, double y) { return sin(x)*cos(y); }
double right( double x, double y) { return sin(y)*cos(x); } 
double jacobian( double x, double y) 
{
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y); 
}
//![function]
double variationRHS( double x, double y) 
{
    return cos(x)*cos(y)*cos(x)*cos(y) + sin(x)*sin(y)*sin(x)*sin(y); 
}
/*
////These are for comparing to FD arakawa results
//double left( double x, double y) {return sin(2.*M_PI*(x-hx/2.));}
//double right( double x, double y) {return y;}
//double jacobian( double x, double y) {return 2.*M_PI*cos(2.*M_PI*(x-hx/2.));}
const double lx = M_PI/2.;
const double ly = M_PI/2.;
double left( double x, double y) {return sin(x)*sin(y);}
double right( double x, double y) {return sin(2*x)*sin(2*y);} 
dg::bc bcx = dg::DIR_NEU; 
dg::bc bcy = dg::DIR_NEU;
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y) 
{
    return cos(x)*sin(y)*2*sin(2*x)*cos(2*y)-sin(x)*cos(y)*2*cos(2*x)*sin(2*y);
}
*/

int main()
{
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;

    //![doxygen]
    const dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    const dg::DVec lhs = dg::evaluate( left, grid);
    const dg::DVec rhs = dg::evaluate( right, grid);
    dg::DVec jac(lhs);

    dg::ArakawaX<dg::aGeometry2d, dg::DMatrix, dg::DVec> arakawa( grid);
    arakawa( lhs, rhs, jac);
    //![doxygen]

    exblas::udouble res;
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec eins = dg::evaluate( dg::one, grid);
    const dg::DVec sol = dg::evaluate ( jacobian, grid);
    res.d = dg::blas2::dot( eins, w2d, jac);
    std::cout << "Mean     Jacobian is "<<res.d<<"\t"<<res.i<<"\n";
    res.d = dg::blas2::dot( rhs, w2d, jac);
    std::cout << "Mean rhs*Jacobian is "<<res.d<<"\t"<<res.i<<"\n";
    res.d = dg::blas2::dot( lhs, w2d, jac);
    std::cout << "Mean lhs*Jacobian is "<<res.d<<"\t"<<res.i<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    res.d = sqrt(dg::blas2::dot( w2d, jac));
    std::cout << "Distance to solution "<<res.d<<"\t"<<res.i<<std::endl; //don't forget sqrt when comuting errors
    //periocid bc       |  dirichlet bc
    //n = 1 -> p = 2    |     
    //n = 2 -> p = 1    |
    //n = 3 -> p = 3    |        3
    //n = 4 -> p = 3    | 
    //n = 5 -> p = 5    |
    // quantities are all conserved to 1e-15 for periodic bc
    // for dirichlet bc these are not better conserved than normal jacobian
    const dg::DVec variation = dg::evaluate ( variationRHS, grid);
    arakawa.variation( rhs, jac);
    dg::blas1::axpby( 1., variation, -1., jac);
    std::cout << "Variation distance to solution "<<sqrt( dg::blas2::dot( w2d, jac))<<std::endl; //don't forget sqrt when comuting errors
    return 0;
}
