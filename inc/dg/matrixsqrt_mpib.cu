#define SILENT

#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "blas.h"
#include "backend/typedefs.h"
#include "topology/evaluation.h"
#include "backend/timer.h"
#include "backend/mpi_init.h"
#include "matrixsqrt.h"
#include "matrixfunction.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
const double alpha = -1.;
const double m=1.;
const double n=1.;
double lhs( double x, double y){ return sin(x*m)*sin(y*n);}
double rhsHelmholtz( double x, double y){ return (1.-(m*m+n*n)*alpha)*sin(x*m)*sin(y*n);}
double rhsHelmholtzsqrt( double x, double y){ return sqrt(1.-(m*m+n*n)*alpha)*sin(x*m)*sin(y*n);}

using Matrix = dg::MDMatrix;
using Container = dg::MDVec;

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    
    dg::Timer t;
    
    unsigned n, Nx, Ny;
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
    double epsCG = 1e-8;
    double epsTrel = 1e-9;
    double epsTabs = 1e-12;
    if(rank==0) std::cout << "# Type epsilon for CG (1e-8), and eps_rel (1e-9) and eps_abs (1e-12) for TimeStepper\n";
    if(rank==0) std::cin >> epsCG >> epsTrel >> epsTabs;
    unsigned max_iter = 1;
    unsigned max_iterC = 1;
    if(rank==0) std::cout << "# Type max_iter of tridiagonalization (500) and of Cauchy integral (40) ?\n";
    if(rank==0) std::cin >> max_iter >> max_iterC;    
    if(rank==0) std::cout << "# Type in eps of tridiagonalization (1e-7)\n";
    double eps = 1e-7; //# of pcg iter increases very much if
    if(rank==0) std::cin >> eps;
    if(rank==0) 
    {
        std::cout <<"# You typed\n"
              <<"epsCG:  "<<epsCG<<"\n"
              <<"epsTrel: "<<epsTrel<<"\n"
              <<"epsTabs: "<<epsTabs<<"\n"
              <<"max_iter: "<<max_iter<<"\n"
              <<"max_iterC: "<<max_iterC<<"\n"
              <<"eps: "<<eps<<std::endl;
    }
    
    double erel = 0;
    std::array<unsigned,2> iter_arr;

    dg::RealCartesianMPIGrid2d<double> g( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy, comm);
   //start and end vectors
    Container x = dg::evaluate(lhs, g);
    Container x_exac = dg::evaluate(lhs, g);
    Container b = dg::evaluate(rhsHelmholtzsqrt, g), b_exac(b), error(b_exac);
    Container bs = dg::evaluate(rhsHelmholtz, g), bs_exac(bs);
    
    const Container w2d = dg::create::weights( g);
    const Container v2d = dg::create::inv_weights( g);

    dg::Helmholtz<dg::aRealMPIGeometry2d<double>, Matrix, Container> A( g, alpha, dg::centered); //not_normed
    dg::Invert<Container> invert( x, g.size(), epsCG);

    double hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
    double max_weights =   dg::blas1::reduce(A.weights(), 0., dg::AbsMax<double>() );
    double min_weights =  -dg::blas1::reduce(A.weights(), max_weights, dg::AbsMin<double>() );
    double kappa = sqrt(max_weights/min_weights); //condition number of weight matrix
    ////////////////////////Direct Cauchy integral solve
    {
        if(rank==0) std::cout << "\nCauchy: \n";
        dg::DirectSqrtCauchySolve<dg::aRealMPIGeometry2d<double>, Matrix, Container> directsqrtcauchysolve(A, g, epsCG, max_iterC);

        t.tic();
        iter_arr[1] = directsqrtcauchysolve(b, bs);
        t.toc();   
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));  
        double time = t.diff();

        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iterT: "<<std::setw(3)<<iter_arr[1] << "\n"; 

        if(rank==0) std::cout << "\nCauchy+CG: \n";
        t.tic();
        iter_arr[0] = invert(A,x,bs);
        t.toc();
        time+=t.diff();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n";   
    }
    //////////////////////Direct sqrt ODE solve
    {
        if(rank==0) std::cout << "\nODE:\n";
        dg::DirectSqrtODESolve<dg::aRealMPIGeometry2d<double>, Matrix, Container> directsqrtodesolve(A, g, epsCG, epsTrel, epsTabs);
        t.tic();
        iter_arr[1] = directsqrtodesolve(b, bs);
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));
        double time = t.diff();
        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iterT: "<<std::setw(3)<<iter_arr[1] << "\n"; 
        
        if(rank==0) std::cout << "\nODE+CG:\n";
        t.tic();
        iter_arr[0] = invert(A,x,bs);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac)); 
        time+=t.diff();
        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n";     
        
    }
    
    //////////////////Krylov solve via Lanczos method and Cauchy solve
    {
        if(rank==0) std::cout << "\nM-Lanczos+Cauchy:\n";
        dg::KrylovSqrtCauchySolve<dg::aRealMPIGeometry2d<double>, Matrix, Container> krylovsqrtcauchysolve(A, g, x,  epsCG, max_iter, max_iterC, eps);
        t.tic();
        iter_arr = krylovsqrtcauchysolve(b, bs); 
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
        double time = t.diff();

        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n"; 
        if(rank==0) std::cout << "    iterT: "<<std::setw(3)<<iter_arr[1] << "\n"; 

        if(rank==0) std::cout << "\nM-Lanczos+Cauchy+CG:\n";
        t.tic();
        iter_arr[0] = invert(A,x,bs);
        t.toc();
        time += t.diff();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n";   
    }
    
    //////////////////Krylov solve via Lanczos method and ODE sqrt solve
    {
        if(rank==0) std::cout << "\nM-Lanczos+ODE:\n";  
        dg::KrylovSqrtODESolve<dg::aRealMPIGeometry2d<double>, Matrix, Container> krylovsqrtodesolve(A, g, x,  epsCG, epsTrel, epsTabs, max_iter, eps);
        b = dg::evaluate(rhsHelmholtzsqrt, g);
        t.tic();
        iter_arr = krylovsqrtodesolve(b, bs); //overwrites b
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
        double time = t.diff();

        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n"; 
        if(rank==0) std::cout << "    iterT: "<<std::setw(3)<<iter_arr[1] << "\n"; 
        
        if(rank==0) std::cout << "\nM-Lanczos+ODE+CG:\n";
        t.tic();
        iter_arr[0] = invert(A,x,bs);
        t.toc();
        time += t.diff();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n"; 
    }
    //////////////////Krylov solve via Lanczos method and ODE sqrt solve
    {
        if(rank==0) std::cout << "\nM-Lanczos+EIGEN:\n";  
        //EVs of Helmholtz
        double EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0);
        double EVmax =1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny())); 
        double res_fac = kappa*sqrt(EVmin);
        
        dg::KrylovFuncEigenSolve<Container> krylovfunceigensolve(x,   max_iter);
        b = dg::evaluate(rhsHelmholtzsqrt, g);
        t.tic();
        iter_arr[0] = krylovfunceigensolve(b, bs, dg::SQRT<double>(), A, A.inv_weights(), A.weights(),  eps, res_fac); 
        t.toc();
        dg::blas1::axpby(1.0, bs, -1.0, bs_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, bs_exac));   
        double time = t.diff();

        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n"; 
        
        if(rank==0) std::cout << "\nM-Lanczos+EIGEN+CG:\n";
        t.tic();
        iter_arr[0] = invert(A,x,bs);
        t.toc();
        time += t.diff();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));   
        if(rank==0) std::cout << "    time: "<<time<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n"; 
    }
    //sqrt invert schemes
    {
        if(rank==0) std::cout << "\nM-CG+Cauchy:\n";
        dg::blas1::scal(x, 0.0);  //must be initialized with zero
        dg::KrylovSqrtCauchyinvert<dg::aRealMPIGeometry2d<double>, Matrix,  Container> krylovsqrtcauchyinvert(A, g, x,  epsCG, max_iter, max_iterC, eps);
        t.tic();
        iter_arr = krylovsqrtcauchyinvert( x, b_exac);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        if(rank==0) std::cout << "    time: "<<t.diff()<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n"; 
        if(rank==0) std::cout << "    iterT: "<<std::setw(3)<<iter_arr[1] << "\n"; 
    }
    {
        if(rank==0) std::cout << "\nM-CG+ODE:\n";
        dg::blas1::scal(x, 0.0); //must be initialized with zero
        dg::KrylovSqrtODEinvert<dg::aRealMPIGeometry2d<double>, Matrix, Container> krylovsqrtodeinvert(A, g, x,  epsCG, epsTrel, epsTabs, max_iter, eps);
        t.tic();
        iter_arr= krylovsqrtodeinvert( x, b_exac);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        if(rank==0) std::cout << "    time: "<<t.diff()<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0] << "\n"; 
        if(rank==0) std::cout << "    iterT: "<<std::setw(3)<<iter_arr[1] << "\n"; 
    }
    {
        if(rank==0) std::cout << "\nM-CG+EIGEN:\n";
        //EVs of inverse Helmholtz
        double EVmin = 1./(1.-A.alpha()*hxhy*(1.0 + 1.0));
        double EVmax = 1./(1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()))); 
        double res_fac = kappa*sqrt(EVmin);
        
        dg::blas1::scal(x, 0.0); //must be initialized with zero
        dg::KrylovFuncEigenInvert< Container> krylovfunceigeninvert( x, max_iter);
        t.tic();
        iter_arr[0]  = krylovfunceigeninvert( x, b_exac, dg::SQRT<double>(), A, A.inv_weights(), A.weights(),  eps, res_fac);
        t.toc();
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        if(rank==0) std::cout << "    time: "<<t.diff()<<"s \n"; 
        if(rank==0) std::cout << "    error: "<<erel  << "\n"; 
        if(rank==0) std::cout << "    iter: "<<std::setw(3)<<iter_arr[0]  << "\n"; 
    }
    
    MPI_Finalize();
    return 0;
}
