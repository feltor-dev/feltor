#include <iostream>

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "polarization_init.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;


dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;

double amp = 0.9;
double phi_ana( double x, double y) { return sin(x)*sin(y); }
double dxphi_ana( double x, double y) { return cos(x)*sin(y); }
double dyphi_ana( double x, double y) { return sin(x)*cos(y); }
double lapphi_ana( double x, double y) { return -2.*sin(x)*sin(y); }

struct rho_ana
{
    rho_ana(dg::Cauchy cauchy) :
    m_cauchy(cauchy) {}
    double operator()(double x, double y )const{
        return -1.*(m_cauchy.dx(x,y)*cos(x)*sin(y) + m_cauchy.dy(x,y)*sin(x)*cos(y)- (1.+ m_cauchy(x,y)) * 2.*sin(x)*sin(y))-m_cauchy(x,y);
    }
private:
    dg::Cauchy m_cauchy;
};


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

    dg::CartesianGrid2d grid2d( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    const Container w2d = dg::create::weights( grid2d);

    dg::Cauchy cauchyfunc( lx/2., ly/2., lx/4., ly/4., amp);
    Container chi =  dg::evaluate( cauchyfunc, grid2d);
    dg::blas1::plus(chi, 1.0);
    Container phi = dg::evaluate( phi_ana, grid2d);
    rho_ana rho_anafunc(cauchyfunc);
    Container rho = dg::evaluate( rho_anafunc, grid2d);
    Container x(rho.size(), 0.), temp(rho), error(rho);

    Container dxphi = dg::evaluate( dxphi_ana, grid2d);
    Container dyphi = dg::evaluate( dyphi_ana, grid2d);
    Container lapphi = dg::evaluate( lapphi_ana, grid2d);


    {
        std::cout << "#####ff polarization charge chi initialization test\n";
        //TODO converges very slowly, should not converge that slowly ....
//         {
//             dg::PolChargeN< dg::CartesianGrid2d, Matrix, Container > polN(grid2d, grid2d.bcx(), grid2d.bcy(), dg::centered, 1., false);
//             polN.set_phi(phi);
//             polN.set_dxphi(dxphi);
//             polN.set_dyphi(dyphi);
//             polN.set_lapphi(lapphi);
//
//             double eps = 1e-5;
//             double maxinner  = 30;
//             double maxouter = 10;
//             double restarts = 1000;
//             std::cout << "Type eps (1e-5), maxinner (30), maxouter (10), restart(1000)\n";
//             std::cin >> eps >>maxinner >> maxouter >> restarts;
//             dg::LGMRES <Container> lgmres( x, maxinner, maxouter, restarts);
//             dg::blas1::scal(x, 0.0);
//             dg::blas1::plus(x, 1.0); //x solution must be positive
//             t.tic();
//             unsigned number = lgmres.solve( polN, x, rho, polN.inv_weights(), polN.weights(), eps, 1);
//             t.toc();
//             dg::blas1::axpby( 1., chi, -1., x, error);
//
//             std::cout << " Time: "<<t.diff() << "\n";
//             std::cout << "number of iterations:  "<<number<< std::endl;
//             std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, chi))<<std::endl;
//         }
//         {
//             dg::PolChargeN< dg::CartesianGrid2d, Matrix, Container > polN(grid2d, grid2d.bcx(), grid2d.bcy(), dg::centered, 1, false);
//             polN.set_phi(phi);
//             polN.set_dxphi(dxphi);
//             polN.set_dyphi(dyphi);
//             polN.set_lapphi(lapphi);
//
//             dg::CG <Container> pcg( x,  grid2d.size()*100);
//             double eps = 1e-5;
//             std::cout << "Type eps (1e-5)\n";
//             std::cin >> eps;
//             dg::blas2::symv(polN.weights(), rho, temp);
//             dg::blas1::scal(x, 0.0);
//             dg::blas1::plus(x, 1.0); //x solution must be positive
//             t.tic();
//             unsigned number = pcg( polN, x, temp, polN.precond(), polN.weights(), eps, 1);
//             t.toc();
//             dg::blas1::axpby( 1., chi, -1., x, error);
//
//             std::cout << " Time: "<<t.diff() << "\n";
//             std::cout << "number of iterations:  "<<number<< std::endl;
//             std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, chi))<<std::endl;
//         }
        {
            dg::mat::PolChargeN< dg::CartesianGrid2d, Matrix, Container >
                polN(grid2d, grid2d.bcx(), grid2d.bcy(), dg::centered, 1.0);
            polN.set_phi(phi);
            polN.set_dxphi(dxphi);
            polN.set_dyphi(dyphi);
            polN.set_lapphi(lapphi);

            double eps = 1e-5;
            double damping = 1e-9;
            unsigned restart = 10000;
            std::cout << "Type eps (1e-5), damping (1e-9) and restart (10000) \n";
            std::cin >> eps >> damping >> restart;
            dg::AndersonAcceleration<Container> acc( x, restart);

            dg::blas1::scal(x, 0.0);
            dg::blas1::plus(x, 1.0); //x solution must be positive

            t.tic();
            unsigned number = acc.solve( polN, x, rho, w2d, eps, eps, grid2d.size()*100, damping, restart, true);
            t.toc();

            dg::blas1::axpby( 1., chi, -1., x, error);

            std::cout << " Time: "<<t.diff() << "\n";
            std::cout << "number of iterations:  "<<number<< std::endl;
            std::cout << "rel error " << sqrt( dg::blas2::dot( w2d, error)/ dg::blas2::dot( w2d, chi))<<std::endl;
        }
//
                    //Plot into netcdf file
        size_t start = 0;
        dg::file::NC_Error_Handle err;
        int ncid;
        err = nc_create( "visual.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
        int dim_ids[3], tvarID;
        err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid2d);

        std::string names[3] = {"sol", "ana", "error"};
        int dataIDs[3];
        for( unsigned i=0; i<3; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

        dg::HVec transferH(dg::evaluate(dg::zero, grid2d));

        dg::assign( x, transferH);
        dg::file::put_vara_double( ncid, dataIDs[0], start, grid2d, transferH);
        dg::assign( chi, transferH);
        dg::file::put_vara_double( ncid, dataIDs[1], start, grid2d, transferH);
        dg::assign( error, transferH);
        dg::file::put_vara_double( ncid, dataIDs[2], start, grid2d, transferH);
        err = nc_close(ncid);


    }

    return 0;
}
