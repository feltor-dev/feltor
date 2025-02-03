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

double left( double x, double y) {
    return sin(x)*cos(y);
}
double right( double x, double y) {
    return sin(y)*cos(x); 
}
double jacobian( double x, double y) {
    return cos(x)*cos(y)*cos(x)*cos(y) - sin(x)*sin(y)*sin(x)*sin(y);
}
//![function]
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
    std::cout<<"This program tests the execution of the arakawa scheme! A test is passed if the number in the second column shows exactly zero!\n";
    unsigned n = 3, Nx = 32, Ny = 48;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;

    //![doxygen]
    // create a Cartesian grid on the domain [0,lx]x[0,ly]
    const dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);

    // evaluate left and right hand side on the grid
    const dg::DVec lhs = dg::construct<dg::DVec>( dg::evaluate( left, grid));
    const dg::DVec rhs = dg::construct<dg::DVec>( dg::evaluate( right, grid));
    dg::DVec jac(lhs);

    // create an Arakawa object
    dg::ArakawaX<dg::aGeometry2d, dg::DMatrix, dg::DVec> arakawa( grid);

    //apply arakawa scheme
    arakawa( lhs, rhs, jac);
    //![doxygen]

    int64_t binary[] = {4358628400772939776,4360428067382886400,4362477496701026304,4562674804459845067,4552797036354693398};
    dg::exblas::udouble res;
    dg::DVec w2d = dg::create::weights( grid);
    //dg::DVec eins = dg::evaluate( dg::one, grid);
    const dg::DVec sol = dg::evaluate ( jacobian, grid);

    res.d = dg::blas2::dot( 1., w2d, jac);
    std::cout << "Mean     Jacobian is "<<res.d<<"\t"<<res.i-binary[0]<<"\n";
    res.d = dg::blas2::dot( rhs, w2d, jac);
    std::cout << "Mean rhs*Jacobian is "<<res.d<<"\t"<<res.i-binary[1]<<"\n";
    res.d = dg::blas2::dot( lhs, w2d, jac);
    std::cout << "Mean lhs*Jacobian is "<<res.d<<"\t"<<res.i-binary[2]<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    res.d = sqrt(dg::blas2::dot( w2d, jac)); //don't forget sqrt when computing errors
    std::cout << "Distance to solution "<<res.d<<"\t\t"<<res.i-binary[3]<<std::endl;
    //periocid bc       |  dirichlet bc
    //n = 1 -> p = 2    |
    //n = 2 -> p = 1    |
    //n = 3 -> p = 3    |        3
    //n = 4 -> p = 3    |
    //n = 5 -> p = 5    |
    // quantities are all conserved to 1e-15 for periodic bc
    // for dirichlet bc these are not better conserved than normal jacobian
    std::cout << "\nContinue with topology/average_t.cu !\n\n";
    return 0;
}
