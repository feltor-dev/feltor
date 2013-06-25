#include <iostream>
#include <iomanip>
#include <GL/glfw.h>
#include <sstream>
#include <omp.h>

#include "toefl/toefl.h"
#include "file/read_input.h"
#include "utility.h"
#include "particle_density.h"
#include "dft_dft_solver.h"
//#include "drt_dft_solver.h"
#include "blueprint.h"

using namespace std;
using namespace toefl;

const unsigned n = 3;
typedef DFT_DFT_Solver<n> Sol;
typedef typename Sol::Matrix_Type Mat;
    
unsigned N; //initialized by init function
double amp, imp_amp; //
double blob_width;
const double slit = 1./500.; //half distance between pictures in units of width
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

Blueprint read( char const * file)
{
    std::cout << "Reading from "<<file<<"\n";
    std::vector<double> para;
    try{ para = file::read_input( file); }
    catch (Message& m) 
    {  
        m.display(); 
        throw m;
    }
    Blueprint bp( para);
    amp = para[10];
    imp_amp = para[14];
    N = para[19];
    omp_set_num_threads( para[20]);
    blob_width = para[21];
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    return bp;
}

    
// The solver has to have the getField( target) function returing M
// and the blueprint() function
template<class Solver>
void drawScene( const Solver& solver, target t)
{
    glClear(GL_COLOR_BUFFER_BIT);
    ParticleDensity particle( solver.getField( TL_IMPURITIES), solver.blueprint());
    double max;
    const typename Solver::Matrix_Type * field;

    if( t == TL_ALL)
    {
        { //draw electrons
        field = &solver.getField( TL_ELECTRONS);
        max = abs_max( *field);
        drawTexture( *field, max, -1.0, -slit, slit*field_ratio, 1.0);
        window_str << scientific;
        window_str <<"ne / "<<max<<"\t";
        //Draw a textured quad
        //upper left
        }

        { //draw Ions
        typename Solver::Matrix_Type ions = solver.getField( TL_IONS);
        particle.linear( ions, solver.getField( TL_POTENTIAL), ions, 0 );
        //upper right
        drawTexture( ions, max, slit, 1.0, slit*field_ratio, 1.0);
        window_str <<" ni / "<<max<<"\t";
        }

        if( solver.blueprint().isEnabled( TL_IMPURITY))
        {
            typename Solver::Matrix_Type impurities = solver.getField( TL_IMPURITIES);
            particle.linear( impurities, solver.getField(TL_POTENTIAL), impurities, 0 );
            max = abs_max( impurities);
            //lower left
            drawTexture( impurities, max, -1.0, -slit, -1.0, -slit*field_ratio);
            window_str <<" nz / "<<max<<"\t";
        }

        { //draw potential
        //field = &solver.getField( TL_POTENTIAL); 
        typename Solver::Matrix_Type phi = solver.getField( TL_POTENTIAL);
        particle.laplace( phi );
        max = abs_max(phi);
        //lower right
        drawTexture( phi, max, slit, 1.0, -1.0, -slit*field_ratio);
        window_str <<" phi / "<<max<<"\t";
        }
    }
    else
    {
        field = &solver.getField( t);
        max = abs_max( *field);
        drawTexture( *field, max, -1.0, 1.0, -1.0, 1.0);
        window_str << scientific;
        window_str <<"Max "<<max<<"\t";
    }

        
}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    Blueprint bp_mod;
    if( argc == 1)
    {
        bp_mod = read("blobs.in");
    }
    else if( argc == 2)
    {
        bp_mod = read( argv[1]);
    }
    else
    {
        cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Blueprint bp = bp_mod;

    field_ratio = bp.boundary().lx/bp.boundary().ly;
    
    bp.display(cout);
    if( bp.boundary().bc_x != TL_PERIODIC)
    {
        cerr << "Only periodic boundaries allowed!\n";
        return -1;
    }
    //construct solvers 
    Sol solver( bp);

    const Algorithmic& alg = bp.algorithmic();
    Mat ne{ alg.ny, alg.nx, 0.}, nz{ ne}, phi{ ne};
    // place some gaussian blobs in the field
    try{
        //init_gaussian( ne, 0.1,0.2, 10./128./field_ratio, 10./128., amp);
        //init_gaussian( ne, 0.1,0.4, 10./128./field_ratio, 10./128., -amp);
        //init_gaussian( ne, 0.5,0.5, 10./128./field_ratio, 10./128., amp);
        init_gaussian( ne, 0.2,0.5, blob_width/bp.boundary().lx, blob_width/bp.boundary().ly, amp);
        //init_gaussian( ne, 0.1,0.8, 10./128./field_ratio, 10./128., -amp);
        //init_gaussian( ni, 0.1,0.5, 0.05/field_ratio, 0.05, amp);
        if( bp.isEnabled( TL_IMPURITY))
        {
            //init_gaussian( nz, 0.5,0.5, 0.05/field_ratio, 0.05, -imp_amp);
            init_gaussian_column( nz, 0.4, 0.025/field_ratio, imp_amp);
        }
        std::array< Mat, n> arr{{ ne, nz, phi}};
        //now set the field to be computed
        solver.init( arr, TL_IONS);
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
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glClearColor(0.f, 0.f, 0.f, 0.f);

    double t = 3*alg.dt;
    Timer timer;
    Timer overhead;
    cout<< "HIT ESC to terminate program \n"
        << "HIT S   to stop simulation \n"
        << "HIT R   to continue simulation!\n";
    target targ = TL_ALL;
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
        if( glfwGetKey( '1')) targ = TL_ELECTRONS;
        else if( glfwGetKey( '2')) targ = TL_IONS;
        else if( glfwGetKey( '3')) targ = TL_IMPURITIES;
        else if( glfwGetKey( '4')) targ = TL_POTENTIAL;
        else if( glfwGetKey( '0')) targ = TL_ALL;
        drawScene(solver, targ);
        window_str << setprecision(2) << fixed;
        window_str << " &&   time = "<<t;
        glfwSetWindowTitle( (window_str.str()).c_str() );
        window_str.str("");
        glfwSwapBuffers();
        timer.tic();
        for(unsigned i=0; i<N; i++)
        {
            solver.step();
            t+= alg.dt;
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
