#include <iostream>
#include <iomanip>
#include <sstream>
#include <thrust/remove.h>
#include <thrust/host_vector.h>


#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"
#include "dg/file/json_utilities.h"

#ifdef OPENGL_WINDOW
#include "draw/host_window.h"
#endif

#include "ns.h"
#include "parameters.h"

using namespace std;
using namespace dg;

#ifdef LOG_POLAR
    typedef dg::geo::LogPolarGenerator Generator;
#else
    typedef dg::geo::PolarGenerator Generator;
#endif

// simple, text based, equidistant output format that makes it easy to compare
// with output from other equidistant codes, etc.
void write(string prefix, const DVec& y0, double time, const Parameters& p, dg::geo::CurvilinearGrid2d& grid)
{
    // interpolate to an equidistant grid
    int N = 1024;
    HVec x(N*N), y(N*N);
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++) {
            x[j*N + i] = i*(p.r_max-p.r_min)/double(N);
            y[j*N + i] = j*2*M_PI/double(N);
        }
    }

    DVec eq(N*N);
    thrust::fill(begin(eq), end(eq),1.0);
    IDMatrix interp_matrix = dg::create::interpolation(x, y, grid, dg::DIR);
    dg::blas2::gemv(interp_matrix, y0, eq);

    // filename
    stringstream fn;
    fn << prefix << "-T" << time << "-n" << p.n << "-Nx" << p.Nx << "-Ny" << p.Ny << ".txt";

    // write an output file
    ofstream fs(fn.str());
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++)
            fs << eq[i*N + j] << " ";
        fs << endl;
    }
}

double u0(double x, double y) {
    double rc = 6.0;
    double epsilon = 0.1;
    double l = 7.0;

    double r = x + 1.0;

    return (1.0+epsilon*cos(l*y))*exp(-2.0*pow(r-rc,2));
}

double u1(double x, double y) {
    double r = x + 1.0;
    return exp(-2.0*pow(r-6.0,2));
}

double u2(double x, double y) {
    return cos(y);
}

double u3(double x, double y) {
    double r = x + 1.0;
    return (4*u1(x,y)*(-6.0+r)*sin(y))/r;
}

double u0_test(double x, double y) {
    double r = x + 1.0;
    return -exp(-pow(r-6.0,2))*(-1+12.0*r+140.0*pow(r,2)-48.0*pow(r,3)+4*pow(r,4))*cos(y)/pow(r,2);
}

double u0_ref(double x, double y) {
    double r = x + 1.0;
    return exp(-pow(r-6.0,2))*cos(y);
}


int main(int argc, char* argv[])
{
    Timer t;
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js;
    if( argc == 1)
        dg::file::file2Json( "input.json", js, dg::file::comments::are_discarded);
    else if( argc == 2)
        dg::file::file2Json( argv[1], js, dg::file::comments::are_discarded);
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Parameters p( js);
    p.display( std::cout);

    Generator generator(p.r_min, p.r_max); // Generator is defined by the compiler
    dg::geo::CurvilinearGrid2d grid( generator, p.n, p.Nx, p.Ny, dg::DIR, dg::PER);

    DVec w2d( create::volume(grid));

#ifdef OPENGL_WINDOW
    //create CUDA context that uses OpenGL textures in Glfw window
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
    draw::RenderHostData render( 1,1);
#endif

    HVec omega = evaluate ( u0, grid);
#ifdef LOG_POLAR
    DVec stencil = evaluate( one, grid);
#else
    DVec stencil = evaluate( LinearX(1.0, p.r_min), grid);
#endif
    DVec y0( omega ), y1( y0);

    //make solver and stepper
    polar::Explicit<dg::geo::CurvilinearGrid2d, DMatrix, DVec> shu( grid, p.eps);
    polar::Diffusion<dg::geo::CurvilinearGrid2d, DMatrix, DVec> diffusion( grid, p.nu);
    dg::ImExMultistep_s< DVec > karniadakis( "ImEx-BDF-3-3", y0, y0.size(), p.eps_time);


    // Some simple tests to see if everything is in order
    DVec _u1 = evaluate(u1,grid);
    DVec _u2 = evaluate(u2,grid);
    DVec _ref = evaluate(u3,grid);
    DVec _cmp = evaluate(u3,grid);
    shu.arakawa()( _u1, _u2, y1);
    double e_arakawa = 0.0;
    for(size_t i=0;i<y1.size();i++) {
        e_arakawa = max(e_arakawa,abs(y1[i]-_ref[i]));
        _cmp[i] = y1[i]-_ref[i];
    }
    cout << "err arakawa: " << e_arakawa << endl;

    DVec u0test = evaluate(u0_test,grid);
    DVec potref = evaluate(u0_ref,grid);
    shu( 0., u0test, y1);
    DVec pot = shu.potential();
    double e=0.0;
    for(size_t i=0;i<pot.size();i++) {
        e = max(e,abs(pot[i]-potref[i]));
        _cmp[i] = pot[i]-potref[i];
    }
    cout << "err pot: " << e << endl;


    t.tic();
    shu(0., y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";


    double vorticity = blas2::dot( stencil , w2d, y0);
    DVec ry0(stencil);
    blas1::pointwiseDot( stencil, y0, ry0);
    double enstrophy = 0.5*blas2::dot( ry0, w2d, y0);
    double energy =    0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    cout << "Total vorticity:  "<<vorticity<<"\n";
    cout << "Total enstrophy:  "<<enstrophy<<"\n";
    cout << "Total energy:     "<<energy<<"\n";

    double time = 0;

    write("equi", y0, time, p, grid);

#ifdef OPENGL_WINDOW
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
#endif

    karniadakis.init( shu, diffusion, time, y0, p.dt);

    t.tic();
    int step = 0;
    while (time < p.maxout*p.itstp*p.dt)
    {
#ifdef OPENGL_WINDOW
        if(glfwWindowShouldClose(w))
            break;

        dg::blas2::symv( equidistant, y0, visual);
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        dg::assign( visual, hvisual);
        render.renderQuad( hvisual, p.n*p.Nx, p.n*p.Ny, colors);
        title << "Time "<<time<< " \ttook "<<t.diff()/(double)p.itstp<<"\t per step";
        glfwSetWindowTitle(w, title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers(w);
#endif

        if(step % ((p.maxout*p.itstp)/10) == 0)
            write("equi", y0, time, p, grid);
        step++;

        //step 
        for( unsigned i=0; i<p.itstp; i++)
        {
            karniadakis.step( shu, diffusion, time, y0 );
        }
        cout << "t=" << time << endl;
    }
    t.toc();

#ifdef OPENGL_WINDOW
    glfwTerminate();
#endif

    write("equi", y0, time, p, grid);

    double vorticity_end = blas2::dot( stencil , w2d, y0);
    blas1::pointwiseDot( stencil, y0, ry0);
    double enstrophy_end = 0.5*blas2::dot( ry0, w2d, y0);
    double energy_end    = 0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    cout << "Vorticity error :  "<<vorticity_end-vorticity<<"\n";
    cout << "Enstrophy error :  "<<(enstrophy_end-enstrophy)<<"\n";
    cout << "Energy error    :  "<<(energy_end-energy)<<"\n";
    cout << "Energy: " << energy << endl;

    cout << "Runtime: " << t.diff() << endl;

    return 0;
}

