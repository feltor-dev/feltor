#include <iostream>

#include <thrust/extrema.h>
#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "solovev.h"
//#include "taylor.h"
//#include "ribeiroX.h"
#include "curvilinearX.h"
#include "separatrix_orthogonal.h"
#include "testfunctors.h"

//const char* parameters = "geometry_params_Xpoint_taylor.json";
const char* parameters = "geometry_params_Xpoint.json";

int main(int argc, char**argv)
{
    std::cout << "Type n, Nx (fx = 1./4.), Ny (fy = 1./22.)\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;
    std::cout << "Type psi_0 (-15)! \n";
    double psi_0, psi_1;
    std::cin >> psi_0;
    auto js = dg::file::file2Json( argc == 1 ? parameters : argv[1]);
    //dg::geo::taylor::Parameters gp(js);
    //dg::geo::TokamakMagneticField c = dg::geo::createTaylorField(gp);
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
    //gp.display( std::cout);
    dg::Timer t;
    //std::cout << "Constructing grid ... \n";
    t.tic();

    ////////////////construct Generator////////////////////////////////////
    //std::cout << "Psi min "<<c.psip()(gp.R_0, 0)<<"\n";
    double R0 = gp.R_0, Z0 = 0;
    //double R_X = gp.R_0-1.4*gp.triangularity*gp.a;
    //double Z_X = -1.0*gp.elongation*gp.a;
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    /////////////no monitor
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi;
    ////////////const monitor
    dg::geo::CylindricalSymmTensorLvl1 monitor_chi = make_Xconst_monitor( c.get_psip(), R_X, Z_X) ;
    /////////////monitor bumped around X-point
    //double radius;
    //std::cout << "Type radius\n";
    //std::cin >> radius;
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi = make_Xbump_monitor( c.get_psip(), R_X, Z_X, radius, radius) ;
    /////////////
    dg::geo::SeparatrixOrthogonal generator(c.get_psip(), monitor_chi, psi_0, R_X,Z_X, R0, Z0,0);
    dg::geo::CurvilinearGridX2d g2d( generator, 0.25, 1./22., n, Nx, Ny, dg::DIR, dg::NEU);
    dg::Elliptic<dg::geo::CurvilinearGridX2d, dg::Composite<dg::DMatrix>, dg::DVec> pol( g2d, dg::forward);
    double fx = 0.25;
    psi_1 = -fx/(1.-fx)*psi_0;
    std::cout << "psi_0 = "<<psi_0<<" psi_1 = "<<psi_1<<"\n";

    t.toc();
    //std::cout << "Construction took "<<t.diff()<<"s\n";
    std::cout << "Computing on "<<n<<" x "<<Nx<<" x "<<Ny<<"\n";
    ///////////////////////////////////////////////////////////////////////////
    dg::file::NcFile file( "testX.nc", dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"}}, g2d.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"}}, g2d.abscissas(1));

    file.defput_var( "xc", {"y", "x"}, {}, {g2d.grid()}, g2d.map()[0]);
    file.defput_var( "yc", {"y", "x"}, {}, {g2d.grid()}, g2d.map()[1]);

    dg::DVec x =    dg::evaluate( dg::zero, g2d);
    ////////////////////////blob solution////////////////////////////////////////
    //const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 480, -300, 70.,1.), g2d);
    //const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d);
    //const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 480, -300, 70., 1. ), g2d);
    //////////////////////////blob solution on X-point/////////////////////////////
    const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 420, -470, 50.,1.), g2d);
    const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d);
    const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 420, -470, 50., 1. ), g2d);
    ////////////////////////////laplace psi solution/////////////////////////////
    //const dg::DVec b =        dg::pullback( c.laplacePsip);
    //const dg::DVec chi =      dg::evaluate( dg::one, g2d);
    //const dg::DVec solution =     dg::pullback( c.psip, g2d);
    /////////////////////////////Dir/////FIELALIGNED SIN///////////////////
    //const dg::DVec b =    dg::pullback( dg::geo::EllipticXDirNeuM(c, psi_0, psi_1), g2d);
    //dg::DVec chi  =  dg::pullback( dg::geo::Bmodule(c), g2d);
    //dg::blas1::plus( chi, 1e4);
    //const dg::DVec chi =  dg::pullback( dg::ONE(), g2d);
    //const dg::DVec solution = dg::pullback( dg::geo::FuncXDirNeu(c, psi_0, psi_1 ), g2d);
    ////////////////////////////////////////////////////////////////////////////

    const dg::DVec vol2d = dg::create::volume( g2d);
    const dg::DVec inv_vol2d = dg::create::inv_volume( g2d);
    const dg::DVec v2d = dg::create::inv_weights( g2d);
    pol.set_chi( chi);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-11;
    std::cout << "eps \t# iterations error \thxX hyX \thx_max hy_max \ttime/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    dg::PCG<dg::DVec > pcg( x, n*n*Nx*Ny);
    unsigned number = pcg.solve(pol, x,b, pol.precond(), vol2d, eps );
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol2d, error);
    const double norm = dg::blas2::dot( vol2d, solution);
    std::cout << sqrt( err/norm) << "\t";
    ///////////////////////////////////metric//////////////////////
    dg::SparseTensor<dg::DVec> metric = g2d.metric();
    dg::DVec gyy = metric.value(1,1), gxx = metric.value(0,0), vol = dg::tensor::volume(metric);
    dg::blas1::transform( gxx, gxx, dg::SQRT<double>());
    dg::blas1::transform( gyy, gyy, dg::SQRT<double>());
    dg::blas1::pointwiseDot( gxx, vol, gxx);
    dg::blas1::pointwiseDot( gyy, vol, gyy);
    dg::blas1::scal( gxx, g2d.hx());
    dg::blas1::scal( gyy, g2d.hy());
    double hxX = dg::interpolate( dg::xspace, (dg::HVec)gxx, 0., 0., g2d);
    double hyX = dg::interpolate( dg::xspace, (dg::HVec)gyy, 0., 0., g2d);
    std::cout << *thrust::max_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::max_element( gyy.begin(), gyy.end()) << "\t";
    std::cout << hxX << "\t";
    std::cout << hyX << "\t";
    std::cout<<t.diff()/(double)number<<"s"<<std::endl;

    file.defput_var( "error", {"y", "x"}, {}, {g2d.grid()}, error);
    file.defput_var( "num_solution", {"y", "x"}, {}, {g2d.grid()}, x);
    file.defput_var( "ana_solution", {"y", "x"}, {}, {g2d.grid()}, solution);
    file.defput_var( "chi", {"y", "x"}, {}, {g2d.grid()}, chi);
    file.close();


    return 0;
}
