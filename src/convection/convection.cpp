#include <iostream>
#include <iomanip>
#include <GL/glfw.h>
#include <sstream>
#include <omp.h>
#include <vector>

#include "toefl/toefl.h"
#include "file/read_input.h"
#include "utility.h"
#include "convection_solver.h"

using namespace std;
using namespace toefl;

typedef Convection_Solver Solver;
typedef typename Solver::Matrix_Type Matrix_Type;
    
unsigned N; //steps between output
double amp; //Perturbation amplitude
double field_ratio; 
unsigned width = 1400, height = 1080; //initial window width (height will be computed)
stringstream window_str;  //window name

void GLFWCALL WindowResize( int w, int h)
{
    // map coordinates to the whole window
    double win_ratio = (double)w/(double)h;
    GLint ww = (win_ratio<field_ratio) ? w : h*field_ratio ;
    GLint hh = (win_ratio<field_ratio) ? w/field_ratio : h;
    glViewport( 0, 0, (GLsizei) ww, hh);
    width = w;
    height = h;
}

Parameter read( char const * file)
{
    std::cout << "Reading from "<<file<<"\n";
    Parameter p;
    vector<double> para;
    try{ para = file::read_input( file); }
    catch (Message& m) 
    {  
        m.display(); 
        throw m;
    }
    p.P = para[1];
    p.R = para[2];
    p.nu = para[3];
    amp = para[4];
    p.nz = para[5];
    p.nx = para[6];
    p.dt = para[7];
    N = para[8];
    p.bc_z = TL_DST10;
    omp_set_num_threads( para[11]);
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";

    p.lz = 1.;
    p.h = p.lz / (double)p.nz;
    p.lx = (double)p.nx * p.h;
    return p;
}

    

void drawScene( const Solver& solver, target t)
{
    glClear(GL_COLOR_BUFFER_BIT);
    double max;
    const typename Solver::Matrix_Type * field;

    field = &solver.getField( t);
    window_str << scientific;
    switch( t)
    {
        case( TEMPERATURE):
            max = solver.parameter().R;
            drawTemperature( *field, max, -1.0, 1.0, -1.0, 1.0);
            window_str <<"Temperature / "<<max<<"\t";
        break;
        case( VORTICITY):
          max = abs_max( *field);
          drawTexture( *field, max, -1.0, 1.0, -1.0, 1.0);
          window_str <<"Vorticity/ "<<max<<"\t";
        break;
        case( POTENTIAL):
          max = abs_max( *field);
          drawTexture( *field, max, -1.0, 1.0, -1.0, 1.0);
          window_str <<"Potential/ "<<max<<"\t";
        break;
    }

        
}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    Parameter p;
    if( argc == 1)
    {
        p = read("input.txt");
    }
    else if( argc == 2)
    {
        p = read( argv[1]);
    }
    else
    {
        cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    field_ratio = p.lx/ p.lz;
    
    p.display(cout);
    //construct solvers 
    Solver solver( p);

    //init to zero
    Matrix_Type theta( p.nz, p.nx, 0.), vorticity( theta), phi( theta);
    // initialize theta here ...
    init_gaussian( theta, 0.2,0.5, 5./128./field_ratio, 5./128., amp);
    init_gaussian( theta, 0.7,0.3, 5./128./field_ratio, 5./128., amp);
    init_gaussian( theta, 0.9,0.2, 5./128./field_ratio, 5./128., -amp);
    //initialize solver
    try{
        std::array< Matrix_Type,2> arr{{ theta, vorticity}};
        //now set the field to be computed
        solver.init( arr, POTENTIAL);
    }catch( Message& m){m.display();}

    ////////////////////////////////glfw//////////////////////////////
    {
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}

    height = width/field_ratio;
    if( !glfwOpenWindow( width, height,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        cerr << "ERROR: glfw couldn't open window!\n";
    }
    glfwSetWindowSizeCallback( WindowResize);

    glEnable( GL_TEXTURE_2D);
    glfwEnable( GLFW_STICKY_KEYS);
    glfwDisable( GLFW_STICKY_MOUSE_BUTTONS);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glClearColor(0.f, 0.f, 0.f, 0.f);

    double t = 3*p.dt;
    Timer timer;
    Timer overhead;
    cout<< "HIT ESC to terminate program \n"
        << "HIT S   to stop simulation \n"
        << "HIT R   to continue simulation!\n";
    target targ = TEMPERATURE;
    while( running)
    {
        overhead.tic();
        //ask if simulation shall be stopped
        glfwPollEvents();
        if( glfwGetKey( 'S')/*||((unsigned)t%100 == 0)*/) 
        {
            do
            {
                glfwWaitEvents();
            } while( !glfwGetKey('R') && 
                     !glfwGetKey( GLFW_KEY_ESC) && 
                      glfwGetWindowParam( GLFW_OPENED) );
        }
        
        //draw scene
        if( glfwGetKey( '1')) targ = TEMPERATURE;
        else if( glfwGetKey( '2')) targ = VORTICITY;
        else if( glfwGetKey( '3')) targ = POTENTIAL;
        drawScene( solver, targ);
        window_str << setprecision(2) << fixed;
        window_str << " &&   time/1e-3 = "<<t*1000.;
        glfwSetWindowTitle( (window_str.str()).c_str() );
        window_str.str("");
        glfwSwapBuffers();
        if( glfwGetMouseButton( GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            int xpos, ypos;
            glfwGetMousePos( &xpos, &ypos); //origin top left, yaxis down
            int width, height;
            glfwGetWindowSize( &width, &height); //origin top left, yaxis down
            double x0 = (double)xpos/(double)width;
            double y0 = (1.-(double)ypos/(double)height);
            solver.setHeat(x0, y0, 5./128./field_ratio, 5./128., amp/10.);
        }
        else if ( glfwGetMouseButton( GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS )
        {
            int xpos, ypos;
            glfwGetMousePos( &xpos, &ypos); //origin top left, yaxis down
            int width, height;
            glfwGetWindowSize( &width, &height); //origin top left, yaxis down
            double x0 = (double)xpos/(double)width;
            double y0 = (1.-(double)ypos/(double)height);
            solver.setHeat(x0, y0, 5./128./field_ratio, 5./128., -amp/10.);
        }
        else
        {
            solver.setHeat(0,0,0,0,0);
        }

        timer.tic();
        for(unsigned i=0; i<N; i++)
        {
            solver.step(); //here is the timestep
            t+= p.dt;
        }
        timer.toc();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
        overhead.toc();
    }
    glfwTerminate();
    cout << "Average time for one step = "<<timer.diff()/(double)N<<"s\n";
    cout << "Overhead for visualisation, etc. per step = "<<(overhead.diff()-timer.diff())/(double)N<<"s\n";
    }
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
    return 0;

}
