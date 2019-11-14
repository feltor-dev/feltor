#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#include "draw/host_window.h"
#include "feltor.h"
#include "implicit.h"
using HVec = dg::HVec;
using DVec = dg::DVec;
using DMatrix = dg::DMatrix;
using IDMatrix = dg::IDMatrix;
using IHMatrix = dg::IHMatrix;
using Geometry = dg::CylindricalGrid3d;
#define MPI_OUT

#include "init.h" //for the source profiles

namespace detail{
template<class Explicit, class Implicit, class Container >
struct FullSystem
{
    FullSystem() = default;
    FullSystem( Explicit exp, Implicit imp, Container temp):
            m_exp( exp), m_imp( imp), m_temp(temp){}

    template<class Container2>
    void operator()( const Container2& y, Container2& yp)
    {
        m_exp( 0, y, m_temp);
        m_imp( 0, y, yp);
        dg::blas1::axpby( 1., m_temp, 1., yp);
    }
    private:
    Explicit m_exp;
    Implicit m_imp;
    Container m_temp;
};
}//namespace detail

template<class Explicit, class Implicit, class Container, class ContainerType2>
void solve_steady_state( Explicit& ex, Implicit& im, Container& x, const Container& b, const ContainerType2& weights, double damping)
{
    Container tmp(x);
    detail::FullSystem<Explicit&, Implicit&, Container&> full(ex,im,tmp);
    // allocate memory
    unsigned mMax =10, restart = mMax;
    dg::AndersonAcceleration<Container> acc( x, mMax);
    // Evaluate right hand side and solution on the grid
    const double eps = 1e-3;
    unsigned max_iter =1000;
    std::cout << "Type maximum iteration number\n";
    std::cin >> max_iter;
    std::cout << "Number of iterations "<< acc.solve( full, x, b, weights, eps, eps, max_iter, damping, restart, true)<<std::endl;

}

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js, gs;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        std::ifstream ks("geometry_params.json");
        is >> js;
        ks >> gs;
    }
    else if( argc == 3)
    {
        std::ifstream is(argv[1]);
        std::ifstream ks(argv[2]);
        is >> js;
        ks >> gs;
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "
                  << argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    js["symmetric"] =  true; //overwrite symmetric parameter
    const feltor::Parameters p( js);
    const dg::geo::solovev::Parameters gp(gs);
    p.display( std::cout);
    gp.display( std::cout);
    /////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    if( p.alpha_mag > 0.)
        mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*mag.psip()(mag.R0(),0.), p.alpha_mag);

    //create RHS
    std::cout << "Constructing Explicit...\n";
    feltor::Explicit<Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag);
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit<Geometry, IDMatrix, DMatrix, DVec> im( grid, p, mag);
    std::cout << "Done!\n";

    bool fixed_profile;
    HVec profile = dg::evaluate( dg::zero, grid);
    HVec source_profile;
    try{
        source_profile = feltor::source_profiles.at(p.source_type)(
        fixed_profile, profile, grid, p, gp, mag);
    }catch ( std::out_of_range& error){
        std::cerr << "Warning: source_type parameter '"<<p.source_type<<"' not recognized! Is there a spelling error? I assume you do not want to continue with the wrong source so I exit! Bye Bye :)\n";
        return -1;
    }

    feltor.set_source( fixed_profile, dg::construct<DVec>(profile), p.omega_source, dg::construct<DVec>(source_profile));

    DVec result = dg::evaluate( dg::zero, grid);
    std::array<std::array<DVec,2>,2> y0;
    y0[0][0] = y0[0][1] =y0[1][0] =y0[1][1] = result;
    std::array<std::array<DVec,2>,2> b(y0);
    DVec weights = dg::create::weights( grid);


    /////////////////////////////////////////////////
    while(true)
    {

    /////////////////////////set up transfer for glfw
    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual), avisual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
    std::map<std::string, const dg::DVec* > v4d;
    v4d["ne-1 / "] = &y0[0][0],               v4d["ni-1 / "] = &y0[0][1];
    v4d["Ue / "]   = &feltor.fields()[1][0],  v4d["Ui / "]   = &feltor.fields()[1][1];
    v4d["Ome / "] = &feltor.potential(0); v4d["Apar / "] = &feltor.induction();
    /////////glfw initialisation ////////////////////////////////////////////
    //
    std::stringstream title;
    std::ifstream is( "window_params.js");
    is >> js;
    is.close();
    unsigned red = js.get("reduction", 1).asUInt();
    double rows = js["rows"].asDouble(), cols = p.Nz/red,
           width = js["width"].asDouble(), height = js["height"].asDouble();
    if ( p.symmetric ) cols = rows, rows = 1;
    GLFWwindow* w = draw::glfwInitAndCreateWindow( cols*width, rows*height, "");
    draw::RenderHostData render(rows, cols);

    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    title << std::setprecision(2) << std::scientific;
    while ( !glfwWindowShouldClose( w ))
    {
        title << std::scientific;
        solve_steady_state( feltor, im, y0, b, weights, p.dt);
        for( auto pair : v4d)
        {
            if(pair.first == "Ome / ")
            {
                dg::assign( feltor.lapMperpP(0), hvisual);
                dg::assign( *pair.second, hvisual);
            }
            else if(pair.first == "ne-1 / " || pair.first == "ni-1 / ")
            {
                dg::assign( *pair.second, hvisual);
                dg::blas1::axpby( 1., hvisual, -1., profile, hvisual);
            }
            else
                dg::assign( *pair.second, hvisual);
            dg::blas2::gemv( equi, hvisual, visual);
            colors.scalemax() = (double)thrust::reduce(
                visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
            colors.scalemin() = -colors.scalemax();
            title <<pair.first << colors.scalemax()<<"   ";
            render.renderQuad( hvisual, grid.n()*grid.Nx(),
                                        grid.n()*grid.Ny(), colors);
        }
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers( w);

    }
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////
    return 0;

}
