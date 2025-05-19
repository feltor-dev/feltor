#include <iostream>
#include <iomanip>


#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "solovev.h"
#include "average.h"
#include "taylor.h"
#include "magnetic_field.h"

int main( int argc, char* argv[])
{
    auto js = dg::file::file2Json( argc == 1 ? "geometry_params_Xpoint.json" : argv[1]);
    const dg::geo::solovev::Parameters gp(js);
    const dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    double R_O = gp.R_0, Z_O = 0.;
    if( !gp.isToroidal() )
        dg::geo::findOpoint( mag.get_psip(), R_O, Z_O);
    const double psipmin = mag.psip()(R_O, Z_O);
    double Rmin=gp.R_0-gp.a;
    double Zmin=-gp.a*gp.elongation;
    double Rmax=gp.R_0+gp.a;
    double Zmax=gp.a*gp.elongation;
    dg::Grid2d grid2d(Rmin,Rmax,Zmin,Zmax, 3,200,200);
    std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
    dg::HVec psipog2d = dg::evaluate( mag.psip(), grid2d);
    // This is a numerical test because the fsa of the curvature operators exactly vanishes (at least for I = cte and almost for everything else)!!
    dg::geo::CylindricalVectorLvl0 gradPsipF = dg::geo::createGradPsip(mag);
    dg::HVec curvNablaBGradPsip = dg::evaluate( dg::geo::ScalarProduct(
                dg::geo::createTrueCurvatureNablaB(mag),
                gradPsipF), grid2d);
    dg::HVec curvKappaKGradPsip = dg::evaluate( dg::geo::ScalarProduct(
                dg::geo::createTrueCurvatureKappa(mag),
                gradPsipF), grid2d);
    dg::HVec gradPsip = dg::evaluate( dg::geo::SquareNorm(gradPsipF, gradPsipF), grid2d);
    ///////////////////Compute flux average////////////////////
    unsigned npsi = 3, Npsi = 64;//set number of psivalues (NPsi % 8 == 0)
    dg::Grid1d grid1d( 0,1,npsi,Npsi, dg::NEU);
    double deltapsi = 0.1;
    double psipmax = dg::blas1::reduce( psipog2d, 0. ,thrust::maximum<double>()); //DEPENDS ON GRID RESOLUTION!!
    if( !gp.isToroidal())
    {
        std::cout << "Compute flux averages\n";
        dg::HVec xpoint_weights = dg::evaluate( dg::cooX2d, grid2d);
        if( gp.hasXpoint() )
        {
            double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
            double Z_X = -1.1*gp.elongation*gp.a;
            dg::geo::findXpoint( mag.get_psip(), R_X, Z_X);
            dg::blas1::pointwiseDot( xpoint_weights,
                    dg::evaluate( dg::geo::ZCutter(Z_X), grid2d), xpoint_weights);
        }
        dg::geo::FluxSurfaceAverage<dg::DVec>  fsa( grid2d, mag, psipog2d, xpoint_weights);
        grid1d = dg::Grid1d (psipmin<psipmax ? psipmin : psipmax, psipmin<psipmax ? psipmax : psipmin, npsi ,Npsi,dg::NEU);
        map1d.emplace_back("psi_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of psi with delta function");
        dg::HVec qprofile;
        //if( gp.equilibrium == "solovev")
        {
            dg::geo::SafetyFactor qprof( mag);
            qprofile = dg::evaluate( qprof, grid1d);
            map1d.emplace_back("q-profile", qprofile,
                "q-profile (Safety factor) using direct integration");
            dg::HVec psit = dg::integrate( qprofile, grid1d);
            map1d.emplace_back("psit1d", psit,
                "Toroidal flux label psi_t integrated  on grid1d using direct q");
            //we need to avoid integrating >=0
            dg::Grid1d g1d_fine(psipmin<0. ? psipmin : 0., psipmin<0. ? 0. : psipmin, npsi ,Npsi,dg::NEU);
            qprofile = dg::evaluate( qprof, g1d_fine);
            dg::HVec w1d = dg::create::weights( g1d_fine);
            double psit_tot = dg::blas1::dot( w1d, qprofile);
            //std::cout << "psit tot "<<psit_tot<<"\n";
            dg::blas1::scal ( psit, 1./psit_tot);
            dg::blas1::transform( psit, psit, dg::SQRT<double>());
            map1d.emplace_back("rho_t", psit,
                "Toroidal flux label rho_t = sqrt( psit/psit_tot) evaluated on grid1d");
        }
        dg::geo::SafetyFactorAverage qprof( grid2d, mag);
        qprofile = dg::evaluate( qprof, grid1d);
        map1d.emplace_back("q-profile_fsa", qprofile,
            "q-profile (Safety factor) using average integration");
        dg::HVec psit = dg::integrate( qprofile, grid1d);
        map1d.emplace_back("psit1d_fsa", psit,
            "Toroidal flux label psi_t integrated on grid1d using average q");
        map1d.emplace_back("psip1d",    dg::evaluate( dg::cooX1d, grid1d),
            "Poloidal flux label psi_p evaluated on grid1d");
        dg::HVec rho = dg::evaluate( dg::cooX1d, grid1d);
        dg::blas1::axpby( -1./psipmin, rho, +1., 1., rho); //transform psi to rho
        map1d.emplace_back("rho", rho,
            "Alternative flux label rho = -psi/psimin + 1");
        dg::blas1::transform( rho, rho, dg::SQRT<double>());
        map1d.emplace_back("rho_p", rho,
            "Alternative flux label rho_p = Sqrt[-psi/psimin + 1]");
        fsa.set_container( (dg::DVec)curvNablaBGradPsip);
        map1d.emplace_back("curvNablaB_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of true Nabla B curvature Dot Grad Psip with delta function");
        fsa.set_container( (dg::DVec)curvKappaKGradPsip);
        map1d.emplace_back("curvKappaK_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of true Kappa curvature Dot Grad Psip with delta function");
        fsa.set_container( (dg::DVec)gradPsip);
        map1d.emplace_back("gradPsip_fsa",   dg::evaluate( fsa,      grid1d),
            "Flux surface average of |Grad Psip| with delta function");

        //other flux labels
        dg::geo::FluxSurfaceIntegral<dg::HVec> fsi( grid2d, mag);
        fsi.set_right( xpoint_weights);
        deltapsi = fsi.get_deltapsi();

        dg::HVec areaT_psip = dg::evaluate( fsi, grid1d);
        dg::HVec w1d = dg::create::weights( grid1d);
        double volumeCoarea = 2.*M_PI*dg::blas1::dot( areaT_psip, w1d);

        //area
        fsi.set_left( gradPsip);
        dg::HVec psi_area = dg::evaluate( fsi, grid1d);
        dg::blas1::scal(psi_area, 2.*M_PI);
        map1d.emplace_back( "psi_area", psi_area,
            "Flux area with delta function");

        dg::geo::FluxVolumeIntegral<dg::HVec> fvi( (dg::CartesianGrid2d)grid2d, mag);
        std::cout << "Delta Rho for Flux surface integrals = "<<-deltapsi/psipmin<<"\n";

        fvi.set_right( xpoint_weights);
        dg::HVec psi_vol = dg::evaluate( fvi, grid1d);
        dg::blas1::scal(psi_vol, 2.*M_PI);
        map1d.emplace_back( "psi_vol", psi_vol,
            "Flux volume with delta function");
        double volumeFVI = 2.*M_PI*fvi(psipmax);
        double volumeSep = 2.*M_PI*fvi(0.);
        std::cout << "volume enclosed by separatrix: "<<volumeSep<<"\n";
        std::cout << "volume test with coarea formula: "<<volumeCoarea<<" "<<volumeFVI
                  <<" rel error = "<<fabs(volumeCoarea-volumeFVI)/volumeFVI<<"\n";
    }
    ///////////Write file
    dg::file::NcFile file("average.nc", dg::file::nc_clobber);
    file.defput_dim( "psi", {{"axis", "X"}}, grid1d.abscissas());
    for(auto tp : map1d)
    {
        file.defput_var( std::get<0>(tp), {"psi"}, {{"long_name",
            std::get<2>(tp)}}, {grid1d}, std::get<1>(tp));
    }
    file.close();
    std::cout << "FILE average.nc CLOSED AND READY TO USE NOW!\n" <<std::endl;

    return 0;
}
