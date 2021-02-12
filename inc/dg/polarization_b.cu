// #define DG_DEBUG
#include <iostream>
#include "blas.h"
#include "dg/file/file.h"

#include "helmholtz.h"
#include "polarization.h"
#include "multigrid.h"
#include "backend/exceptions.h"
#include "multistep.h"
#include "cg.h"
#include "functors.h"
#include "andersonacc.h"

const double tau = 1.;
const double alpha = -tau;
const double m = 4.;
const double n = 4.;

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

// df
double phi_ana_df( double x,double y){ return sin(m*x)*sin(n*y);}
double rho_ana_df( double x,double y){ return (m*m+n*n)/(-1.+(m*m+n*n)*alpha)*sin(m*x)*sin(n*y);}

// full_f 
double amp = 0.1;
double chi_ana( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
// double rho_ana_FF( double x, double y) { return (amp*cos(2.*x)*(1./sqrt(1.-4.*alpha) - 2.*cos(2.*y)/sqrt(1.-8.*alpha)) + amp* cos(2.*y)/sqrt(1.-4.*alpha)-(4.*sin(x)*sin(y))/sqrt(1.-2.*alpha))/2.; }
// double phi_ana_FF(double x, double y)  { return (sin(x)*sin(y))*sqrt(1.-(1.+1.)*alpha);}

double rho_ana_FF( double x, double y) { return (
    amp*cos(2.*x)*(1./sqrt(1.-4.*alpha)  - 2.*cos(2.*y)/sqrt(1.-8.*alpha)) 
    + amp* cos(2.*y)/sqrt(1.-4.*alpha)
    -(4.*sin(x)*sin(y))/sqrt(1.-2.*alpha)
        )/2./sqrt(1.-2.*alpha); }
double phi_ana_FF(double x, double y)  { return (sin(x)*sin(y));}

//Full f cold
double rho_ana_FF_cold( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
double phi_ana_FF_cold(double x, double y)  { return sin( x)*sin(y);}

using DiaMatrix = cusp::dia_matrix<int, double, cusp::device_memory>;
using CooMatrix = cusp::coo_matrix<int, double, cusp::device_memory>;
using Matrix = dg::DMatrix;
using Container = dg::DVec;
using SubContainer = dg::DVec;

int main()
{
    dg::Timer t;

    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n>> Nx >> Ny;
    
    double eps = 1e-6;
    double eps_gamma = 1e-7;
    std::cout << "Type in eps_pol and eps_gamma (eps_gamma < eps_pol)\n";
    std::cin >> eps >> eps_gamma;
    
    dg::Grid2d grid2d( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    const Container w2d = dg::create::weights( grid2d);
    const Container v2d = dg::create::inv_weights( grid2d);
    const Container one = dg::evaluate( dg::one, grid2d);
    const Container rho = dg::evaluate( rho_ana_df, grid2d);
    const Container sol = dg::evaluate( phi_ana_df, grid2d);
    const Container rho_FF = dg::evaluate( rho_ana_FF, grid2d);
    const Container sol_FF = dg::evaluate( phi_ana_FF, grid2d);
    Container x(rho.size(), 0.), temp(rho), error(rho), x_gamma(x);
    const Container chi =  dg::evaluate( chi_ana, grid2d);
    
    exblas::udouble res;
    unsigned number = 0;
    dg::Helmholtz< dg::CartesianGrid2d, Matrix, Container > gamma0inv(  grid2d,grid2d.bcx(),grid2d.bcy(), alpha ,dg::centered);
    dg::Helmholtz< dg::CartesianGrid2d, Matrix, Container > gamma0inv_per(  grid2d, dg::PER, grid2d.bcy(), alpha ,dg::centered);
    dg::Elliptic< dg::CartesianGrid2d, Matrix, Container > lapperp(grid2d, grid2d.bcx(), grid2d.bcy(), dg::not_normed, dg::centered);

//     dg::Invert<Container> invert( x, grid2d.size(), eps, 0, true, 1.);
    dg::CG <Container> pcg( x, grid2d.size());


    dg::PolCharge< dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer > pol_df, pol_ff;
    
    pol_df.construct(alpha, {eps_gamma, 0.1*eps_gamma, 0.1*eps_gamma}, grid2d, grid2d.bcx(), grid2d.bcy(), dg::not_normed, dg::centered, 1., true, "df");
    pol_ff.construct(alpha, {eps_gamma}, grid2d, grid2d.bcx(), grid2d.bcy(), dg::not_normed, dg::centered, 1., true, "ff");
    
//     std::cout << "#####df polarization charge with nested inversion (commute = false)\n";    
//     pol_df.set_commute(false);
//     dg::blas1::scal(x,0.0);   
//     t.tic();
//     dg::blas1::pointwiseDot( pol_df.weights(), rho, temp);
// 
//     number = pcg( pol_df, x, temp, pol_df.precond(), pol_df.weights(), eps);
//             if(  number == pcg.get_max())
//             throw dg::Fail( eps);
//     dg::blas1::scal(x, -1.0);
//     t.toc();
//     
//     dg::blas1::axpby( 1., sol, -1., x, error);
//     res.d = sqrt( dg::blas2::dot( w2d, error));
//     std::cout << " Time: "<<t.diff() << "\n";
//     std::cout << "number of iterations:  "<<number<<std::endl;
//     std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
//     std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
//     
//     std::cout << "#####df polarization charge with nested inversion (commute = true)\n";    
//     pol_df.set_commute(true);
//     dg::blas1::scal(x,0.0);   
//     t.tic();
//     dg::blas1::pointwiseDot( pol_df.weights(), rho, temp);
// 
//     number = pcg( pol_df, x, temp, pol_df.precond(), pol_df.weights(), eps);
//             if(  number == pcg.get_max())
//             throw dg::Fail( eps);
//     dg::blas1::scal(x, -1.0);
//     t.toc();
//     
//     dg::blas1::axpby( 1., sol, -1., x, error);
//     res.d = sqrt( dg::blas2::dot( w2d, error));
//     std::cout << " Time: "<<t.diff() << "\n";
//     std::cout << "number of iterations:  "<<number<<std::endl;
//     std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
//     std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
// 
// 
// 
//     std::cout << "#####df polarization charge without nested inversion (commute = false)\n";    dg::blas1::scal(x,0.0);
//     t.tic();
//     dg::blas2::symv(w2d, rho, temp); //not normed
//     dg::blas1::scal(temp,-1.0);
//     number = pcg( lapperp, x, temp, v2d, w2d, eps);
//             if(  number == pcg.get_max())
//             throw dg::Fail( eps);
//     dg::blas2::symv(gamma0inv, x, temp); 
//     dg::blas2::symv(v2d, temp, x);
//     t.toc();
//     
//     dg::blas1::axpby( 1., sol, -1., x, error);
// 
//     res.d = sqrt( dg::blas2::dot( w2d, error));
//     
//     std::cout << " Time: "<<t.diff() << "\n";
//     std::cout << "number of iterations:  "<<number<<std::endl;
//     std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
//     std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
//     
//     
//     std::cout << "#####df polarization charge without nested inversion (commute = true)\n";  
//     dg::blas1::scal(x, 0.0);
//     t.tic();
//     dg::blas2::symv(gamma0inv, rho, temp); // not normed for cg inversion
//     dg::blas1::scal(temp,-1.0);
//     number = pcg( lapperp, x, temp, v2d, w2d, eps);
//             if(  number == pcg.get_max())
//             throw dg::Fail( eps);
//     t.toc();
//     
//     dg::blas1::axpby( 1., sol, -1., x, error);
// 
//     res.d = sqrt( dg::blas2::dot( w2d, error));
//     
//     std::cout << " Time: "<<t.diff() << "\n";
//     std::cout << "number of iterations:  "<<number<<std::endl;
//     std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
//     std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol))<<std::endl;
    
//     /////////////////Full-F polarization (order 2)    
    std::cout << "#####ff polarization charge with nested inversion (commute = false)\n";
    pol_ff.set_commute(false);
    pol_ff.set_chi(chi);
    //TODO not converging
//     dg::blas1::copy(sol_FF, x);
    dg::blas1::scal(x, 0.0);    

    t.tic();
    dg::blas1::pointwiseDot(w2d, rho_FF, temp);
    number = pcg( pol_ff, x, temp, v2d, w2d, eps);
            if(  number == pcg.get_max())
            throw dg::Fail( eps);
    t.toc();
    dg::blas1::scal(x,-1.0);
    dg::blas1::axpby( 1., sol_FF, -1., x, error);

    res.d = sqrt( dg::blas2::dot( w2d, error));
    
    std::cout << " Time: "<<t.diff() << "\n";
    std::cout << "number of iterations:  "<<number<<std::endl;
    std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol_FF))<<std::endl;
    

  /*  
    //test application of operator //TODO not converging but relatively close to sol
    dg::blas2::symv( pol_ff,  sol_FF, x);
    dg::blas1::pointwiseDot(pol_ff.inv_weights(), x, x);
    dg::blas1::scal(x,-1.0);
    dg::blas1::axpby( 1., rho_FF, -1., x, error);

    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, rho_FF))<<std::endl;*/
    
    

       
/*
    std::cout << "#####ff polarization charge without nested inversion (commute = false)\n";
    lapperp.set_chi( chi);
    KrylovSqrtCauchySolve< dg::CartesianGrid2d, Matrix, DiaMatrix, CooMatrix, Container, SubContainer> sqrtsolve, sqrtsolve_per;
    sqrtsolve.construct( gamma0inv, grid2d, chi,  1e-10, 200, 20,  eps_gamma);
    sqrtsolve_per.construct( gamma0inv_per, grid2d, chi,  1e-10, 200, 20,  eps_gamma);
    
    dg::blas1::scal(x_gamma, 0.0);
    dg::blas1::scal(temp, 0.0);
    dg::blas1::scal(x, 0.0);
    t.tic();
    sqrtsolve_per(rho_FF, temp); 
    dg::blas1::scal(temp,-1.0);
    dg::blas1::pointwiseDot(w2d, temp, temp); //make not normed
    number = pcg( lapperp, x_gamma, temp, v2d, w2d, eps);
            if(  number == pcg.get_max())
            throw dg::Fail( eps);
    sqrtsolve(x_gamma, x);       
    t.toc();
    dg::blas1::axpby( 1., sol_FF, -1., x, error);

    res.d = sqrt( dg::blas2::dot( w2d, error));
    
    std::cout << " Time: "<<t.diff() << "\n";
    std::cout << "number of CG iterations:  "<<number<<std::endl;
    std::cout << "abs error " << res.d<<"\t"<<res.i<<std::endl;
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, sol_FF))<<std::endl;*/
    
    size_t start = 0;
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( "visual.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid2d);
    
    std::string names[3] = {"sol", "ana", "error"};
    int dataIDs[3];
    for( unsigned i=0; i<3; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    dg::HVec transferH(dg::evaluate(dg::zero, grid2d));
    
    dg::blas1::transfer( x, transferH);
    file::put_vara_double( ncid, dataIDs[0], start, grid2d, transferH);
    dg::blas1::transfer( sol_FF, transferH);
    file::put_vara_double( ncid, dataIDs[1], start, grid2d, transferH);
    dg::blas1::transfer( error, transferH);
    file::put_vara_double( ncid, dataIDs[2], start, grid2d, transferH);
    err = nc_close(ncid); 
    
   
/*    
       std::cout << "#####ff polarization charge chi initialization test\n";
//TODO
    dg::PolChargeN< dg::CartesianGrid2d, Matrix, Container > polN(grid2d,grid2d.bcx(), grid2d.bcy(), dg::normed, dg::centered);
    
    Container rho_cold = dg::evaluate(rho_ana_FF_cold, grid2d);
    Container phi_cold = dg::evaluate(phi_ana_FF_cold, grid2d);
    
    dg::AndersonAcceleration<Container> acc( x, 10);
    
    polN.set_phi(phi_cold);
    dg::blas1::scal(x, 0.0);
    dg::blas1::plus(x, 1.0); //x solution must be positive
    
//     dg::blas1::copy(chi, x); //solution as guess works
    double damping;
    unsigned restart;
    std::cout << "Type eps (1e-5), damping (1e-9) and restart (10) \n";
    std::cin >> eps >> damping >> restart;
    std::cout << "Number of iterations "<< acc.solve( polN, x, rho_cold, w2d, eps, eps*eps, grid2d.size()*grid2d.size(), damping,restart, true)<<std::endl;
    dg::blas1::axpby( 1., chi, -1., x, error);

    res.d = sqrt( dg::blas2::dot( w2d, error));
    
    std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, chi))<<std::endl;*/
    
    return 0;
}
