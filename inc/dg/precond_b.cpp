#include <iostream>
#include <iomanip>
#include <cusp/elementwise.h>

#include "dg/algorithm.h"
#include "precond.h"

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

double initial( double x, double y) {return 0.;}
//double amp = 0.9999; // LGMRES has problems here
//double amp = 0.9;
double amp = 0.0;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//double pol( double x, double y) {return 1.; }
//double pol( double x, double y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
//double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}
double derX(double x, double y)  { return cos( x)*sin(y);}
double derY(double x, double y)  { return sin( x)*cos(y);}
double vari(double x, double y)  { return pol(x,y)*pol(x,y)*(derX(x,y)*derX(x,y) + derY(x,y)*derY(x,y));}

int main()
{
    unsigned n, Nx, Ny;
    double eps;
	n = 3;
	Nx = Ny = 64;
	eps = 1e-6;
	std::cout << "Type n, Nx and Ny and epsilon! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps;

    std::cout << "Computation on: "<< n <<" x "<< Nx <<" x "<< Ny << std::endl;
	const dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    //create functions A(chi) x = b
    const dg::DVec chi =  dg::evaluate( pol, grid);
    dg::DVec chi_inv(chi);
    dg::blas1::transform( chi, chi_inv, dg::INVERT<double>());
    //compute error
    const dg::DVec solution = dg::evaluate( sol, grid);
    const dg::DVec b =        dg::evaluate( rhs,     grid);
    const dg::DVec derivati = dg::evaluate( derY, grid);
    const dg::DVec variatio = dg::evaluate( vari, grid);
    const double norm = dg::blas2::dot( w2d, solution);
    dg::DVec error( solution);
    dg::exblas::udouble res;

    dg::IHMatrix dx = dg::create::dy( grid, dg::centered).asCuspMatrix();
    dg::IDMatrix dxD = dx;

    dg::blas2::symv( dxD, solution, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    double errn = dg::blas2::dot( w2d, error);
    const double norm_der = dg::blas2::dot( w2d, derivati);
    std::cout << "L2 Norm of relative error in derivative is\n "<<std::setprecision(16)<< sqrt( errn/norm_der)<<std::endl;



    std::cout << "Centered Elliptic Multigrid\n";
    dg::Timer t;
    t.tic();


    const unsigned stages = 3;

    dg::NestedGrids<dg::aGeometry2d, dg::DMatrix, dg::DVec > multigrid( grid, stages);

    const std::vector<dg::DVec> multi_chi = multigrid.project( chi);
    const std::vector<dg::DVec> multi_chi_inv = multigrid.project( chi_inv);

    std::vector<dg::IDMatrix > multi_pol( stages);
    // SAINV preconditioner
    std::vector<dg::IDMatrix > multi_Z(stages), multi_ZT(stages);
    std::vector<dg::DVec> multi_D(stages);

    std::vector<dg::PCG<dg::DVec> > multi_pcg( stages);
    std::vector<std::function<void( const dg::DVec&, dg::DVec&)> >
        multi_inv_pol(stages), multi_inv_precon_pol(stages), multi_precon(stages);
    double threshold = 1e-3;
    unsigned nnzmax = 10;
    std::cout << "Type nnzmax (10) and threshold (1e-3)\n";
    std::cin >> nnzmax >> threshold;

    for(unsigned u=0; u<stages; u++)
    {
        enum dg::direction dir = dg::centered;
        dg::IHMatrix leftx = dg::create::dx( multigrid.grid(u), dg::inverse( bcx), dg::inverse(dir)).asCuspMatrix();
        dg::blas1::scal( leftx.values, -1.);
        dg::IHMatrix lefty =  dg::create::dy( multigrid.grid(u), dg::inverse( bcy), dg::inverse(dir)).asCuspMatrix();
        dg::blas1::scal( lefty.values, -1.);
        dg::IHMatrix rightx =  dg::create::dx( multigrid.grid(u), bcx, dir).asCuspMatrix();
        dg::IHMatrix righty =  dg::create::dy( multigrid.grid(u), bcy, dir).asCuspMatrix();
        dg::IHMatrix jumpx =  dg::create::jumpX( multigrid.grid(u), bcx).asCuspMatrix();
        dg::IHMatrix jumpy =  dg::create::jumpY( multigrid.grid(u), bcy).asCuspMatrix();
        dg::IHMatrix chi_diag = dg::create::diagonal( (dg::HVec)multi_chi[u]);
        dg::IHMatrix CX, XX, CY, YY, JJ, result;

        cusp::multiply( chi_diag, rightx, CX);
        cusp::multiply( leftx, CX, XX );
        cusp::multiply( chi_diag, righty, CY);
        cusp::multiply( lefty, CY, YY );
        cusp::add( jumpx, jumpy, JJ);
        cusp::add( XX, YY, CX);
        cusp::add( CX, JJ, result);
        multi_pol[u] = result;
        dg::IHMatrix Z;
        dg::HVec D;
        dg::HVec weights = dg::create::weights( multigrid.grid(u));
        dg::Timer t;
        std::cout << "Create preconditioner at stage "<<u<<"\n";
        t.tic();
        cusp::multiply( leftx, rightx, XX );
        cusp::multiply( lefty, righty, YY );
        cusp::add( jumpx, jumpy, JJ);
        cusp::add( XX, YY, CX);
        cusp::add( CX, JJ, result);
        dg::create::sainv_precond( result, Z, D, weights, nnzmax, threshold);
        t.toc();
        std::cout << "Took "<<t.diff()<<"s\n";
        dg::IHMatrix zT, zTD;
        cusp::transpose( Z, zT);
        // create D^{-1}
        dg::blas1::pointwiseDot( D, (dg::HVec)multi_chi[u], D);
        dg::blas1::pointwiseDivide( 1., D, D);
        dg::IHMatrix dinv = dg::create::diagonal( D);
        cusp::multiply( zT, dinv, zTD);
        cusp::multiply( zTD, Z, zT);
        dg::IHMatrix W = dg::create::diagonal( weights);
        cusp::multiply( zT, W, Z);
        dg::IDMatrix Zdevice = Z;


        multi_precon[u] =
            [ Zdevice ]( const dg::DVec& y, dg::DVec& x)
            {
                dg::blas2::gemv( Zdevice, y, x);
            };

        cusp::coo_matrix<int, double, cusp::host_memory> host = result;
        if( u==0)
            std::cout << "Matrix dimensions "<< host.num_rows<<" "<<host.num_cols<<" "<<host.values.size()<<"\n";

        multi_pcg[u].construct( multi_chi[u], 1000);
        multi_inv_pol[u] = [&, u, &pcg = multi_pcg[u], &pol = multi_pol[u], weights = (dg::DVec)dg::create::weights( multigrid.grid(u)), &precon = multi_chi_inv[u]](
            const auto& y, auto& x)
            {
                dg::Timer t;
                t.tic();
                int number;
                if ( u == 0)
                    number = pcg.solve( pol, x, y, precon, weights, eps,
                        1, 1);
                else
                    number = pcg.solve( pol, x, y, precon, weights, 1.5*eps,
                    1, 10);
                t.toc();
                std::cout << "# Nested iterations stage: " << u << ", iter: " << number << ", took "<<t.diff()<<"s\n";
            };
        multi_inv_precon_pol[u] = [&, u, &pcg = multi_pcg[u], &pol = multi_pol[u], weights = (dg::DVec)dg::create::weights( multigrid.grid(u)), &precon = multi_precon[u]](
            const auto& y, auto& x)
            {
                dg::Timer t;
                t.tic();
                int number;
                if ( u == 0)
                    number = pcg.solve( pol, x, y, precon, weights, eps,
                        1, 1);
                else
                    number = pcg.solve( pol, x, y, precon, weights, 1.5*eps,
                    1, 10);
                t.toc();
                std::cout << "# Nested iterations stage: " << u << ", iter: " << number << ", took "<<t.diff()<<"s\n";
            };
    }
    t.toc();

    std::cout << "Creation of multigrid took: "<<t.diff()<<"s\n";
    dg::DVec x       =    dg::evaluate( initial, grid);
    t.tic();
    //nested_iterations(multi_pol, x, b, multi_inv_pol, multigrid);
    nested_iterations(multi_pol, x, b, multi_inv_precon_pol, multigrid);
    t.toc();
    std::cout << "Solution with new  preconditioner took "<< t.diff() <<"s\n";
    x       =    dg::evaluate( initial, grid);
    t.tic();
    nested_iterations(multi_pol, x, b, multi_inv_pol, multigrid);
    //nested_iterations(multi_pol, x, b, multi_inv_precon_pol, multigrid);
    t.toc();
    std::cout << "Solution with diag preconditioner took "<< t.diff() <<"s\n";
    dg::blas1::axpby( 1.,x,-1., solution, error);
    errn = dg::blas2::dot( w2d, error);
    errn = sqrt( errn/norm); res.d = errn;
    std::cout << " "<<errn << "\t"<<res.i<<"\n";
    dg::DMatrix DX = dg::create::dy( grid);
    dg::blas2::gemv( DX, solution, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    errn = dg::blas2::dot( w2d, error);
    std::cout << "L2 Norm of relative error in derivative is\n "<<std::setprecision(16)<< sqrt( errn/norm_der)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2
    //
    x       =    dg::evaluate( initial, grid);
    dg::PCG<dg::DVec> pcg( x, 1000);
    t.tic();
    //nested_iterations(multi_pol, x, b, multi_inv_precon_pol, multigrid);
    //pcg.solve( multi_pol[0], x, b, multi_inv_precon_pol[0], w2d, eps, 1);
    //unsigned number = pcg.solve( multi_pol[0], x, b, multi_chi_inv[0], w2d, eps, 1);
    unsigned number = pcg.solve( multi_pol[0], x, b, multi_precon[0], w2d, eps, 1);
    std::cout << "Number "<<number<< " iterations\n";

    t.toc();
    std::cout << "Solution with new  preconditioner took "<< t.diff() <<"s\n";
    x       =    dg::evaluate( initial, grid);
    t.tic();
    number = pcg.solve( multi_pol[0], x, b, multi_chi_inv[0], w2d, eps, 1);
    std::cout << "Number "<<number<< " iterations\n";

    t.toc();
    std::cout << "Solution with diag preconditioner took "<< t.diff() <<"s\n";
    dg::blas1::axpby( 1.,x,-1., solution, error);
    errn = dg::blas2::dot( w2d, error);
    errn = sqrt( errn/norm); res.d = errn;
    std::cout << " "<<errn << "\t"<<res.i<<"\n";
    dg::blas2::gemv( DX, solution, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    errn = dg::blas2::dot( w2d, error);
    std::cout << "L2 Norm of relative error in derivative is\n "<<std::setprecision(16)<< sqrt( errn/norm_der)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2

    return 0;
}
