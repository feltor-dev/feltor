#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"

#include "convection.cuh"
#include "dg/rk.cuh"
#include "dg/timer.cuh"

#include "file/read_input.h"

struct InitDens
{
    InitDens( double alpha, double zeta): alpha_( alpha), zeta_(zeta){}

    double operator()( double x, double y)
    {
        return tanh( (zeta_-y)/alpha_);
    }
  private:
    double alpha_, zeta_;
};

struct Parameters
{
    unsigned n, k,  Nx, Ny;
    double dt; 
    double eps_lap;
    double lx, ly;

    double eps, R_l, L, P, R, zeta;
    double n0; 
    unsigned itstp;
    unsigned maxout;
    Parameters( const std::vector<double>& v)
    {
        n = (unsigned)v[1];
        k = (unsigned)v[4];
        Nx = (unsigned)v[2];
        Ny = (unsigned)v[3];
        dt = v[5];
        eps_lap = v[6];
        lx = v[7];
        ly = v[8];
        R = v[9];
        P = v[10];
        L = v[11];
        eps = v[12];
        R_l = v[13];
        zeta = v[14];
        itstp = v[15];
        maxout = v[16];
    }
    void display( std::ostream& os = std::cout ) const
    {
        os <<  "Algorithmic parameters are: \n"
           <<  "    n = "<<n<<"\n"
           <<  "    Nx = "<<Nx<<"\n"
           <<  "    Ny = "<<Ny<<"\n"
           <<  "    k = "<<k<<"\n"
           <<  "    dt = "<<dt<<"\n"
           <<  "    eps_lap = "<<eps_lap<<"\n";
        os <<  "Boundary parameters are: \n"
           <<  "    lx = "<<lx<<"\n"
           <<  "    ly = "<<ly<<"\n";
        os <<  "Physical Parameters are: \n"
           <<  "    R = "<<R<<"\n"
           <<  "    P = "<<P<<"\n"
           <<  "    L = "<<L<<"\n"
           <<  "    eps = "<<eps<<"\n"
           <<  "    R_l = "<<R_l<<"\n"
           <<  "    zeta = "<<zeta<<"\n";
        os <<  "Steps between output: "<<itstp<<"\n"
           <<  "Number of outputs:    "<<maxout<<std::endl; //implicit flush!
    }
};

const unsigned k = 3;

int main(int argc, char* argv[])
{
    std::vector<double> v, v2;
    if( argc==1)
        try{ v=file::read_input( "input.txt");}catch( toefl::Message& m){m.display();}
    else if( argc==2)
        try{ v=file::read_input( argv[1]);}catch( toefl::Message& m){m.display();}
    else
    {
        std::cerr << "ERROR: Too many arguments! \n Usage: "<<argv[0]<<" [filename]\n";
        return -1;
    }
    try{ v2=file::read_input( "window_params.txt");}catch( toefl::Message& m){m.display();}
    draw::HostWindow w( v2[3], v2[4]);
    w.set_multiplot( v2[1], v2[2]);

    const Parameters p(v);
    if( p.k != k)
    {
        std::cerr << "ERROR: k doesn't match "<<k<<" vs. "<<p.k<<"\n";
        return -1;
    }
    Params params; 
    params.eps = p.eps, params.P = p.P;
    params.R = p.R, params.L = p.L, params.R_l = p.R_l, params.zeta = p.zeta;

    dg::Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, dg::PER, dg::DIR);
    Convection< dg::DVec> convect( grid, params, p.eps_lap);
    //initial conditions
    dg::Gaussian g1( 0.4*p.lx, 0.3*p.ly, 5, 5, p.n0);
    dg::Gaussian g2( 0.7*p.lx, 0.35*p.ly, 5, 5,  p.n0);
    std::vector<dg::DVec> y0( 3);
    y0[0] = y0[1] = y0[2] = dg::evaluate( g1, grid);
    dg::blas1::axpby( 1., y0[0], 1., (dg::DVec)dg::evaluate( g2, grid), y0[0]);
    y0[1] = dg::evaluate( InitDens(0.001, p.zeta), grid);
    dg::blas1::axpby( 1., y0[2], -1, y0[2]);
    //init timestepper
    dg::AB< k, std::vector<dg::DVec> > ab(y0);
    ab.init( convect, y0, p.dt);
    std::vector<dg::DVec> y1( y0);
    ab( convect, y0, y1, p.dt);
    y0.swap(y1);
    unsigned step = 0;
    double time = 0;
    //visualization
    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    bool running = true;
    while ( running) 
    {
        dg::blas1::axpby( 1., y0[0], 1., convect.background(), dvisual);
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        colors.scale() = p.R/2.;

        w.title() << std::setprecision(2) << std::scientific;
        w.title() <<"temp / "<<colors.scale()<<"\t";
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        //steps
        for( unsigned i=0; i<p.itstp; i++)
        {
            step++;
            try{ ab( convect, y0, y1, p.dt);}
            catch( Fail& fail){
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does simulation respect CFL condition?\n";
                running = false;
                break;
            }
        }
        y0.swap( y1);
        running = running &&
            !glfwGetKey( GLFW_KEY_ESC) &&
            glfwGetWindowParam( GLFW_OPENED);
    }



    
}
