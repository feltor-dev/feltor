#include <iostream>
#include <iomanip>
#include <GL/glfw.h>
#include <sstream>
#include <omp.h>

#include "toefl/toefl.h"
#include "file/read_input.h"
#include "utility.h"
#include "dft_dft_solver.h"
#include "drt_dft_solver.h"
#include "blueprint.h"

/*
 * Reads parameters from given input file
 * Inititalizes the correct solver 
 * visualizes results directly on the screen
 */
using namespace std;
using namespace toefl;
    
unsigned N; //initialized by init function
double amp, imp_amp; //
const double slit = 2./500.; //half distance between pictures in units of width
double field_ratio;
unsigned width = 960, height = 1080; //initial window width & height
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
    field_ratio = bp.boundary().lx/bp.boundary().ly;
    omp_set_num_threads( para[20]);
    //blob_width = para[21];
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    return bp;
}
    

// The solver has to have the getField( target) function returing M
// and the blueprint() function
template<class Solver>
void drawScene( const Solver& solver)
{
    glClear(GL_COLOR_BUFFER_BIT);
    double max;
    const typename Solver::Matrix_Type * field;
    
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
    field = &solver.getField( TL_IONS);
    //upper right
    drawTexture( *field, max, slit, 1.0, slit*field_ratio, 1.0);
    window_str <<" ni / "<<max<<"\t";
    }

    if( solver.blueprint().isEnabled( TL_IMPURITY))
    {
        field = &solver.getField( TL_IMPURITIES); 
        max = abs_max(*field);
        //lower left
        drawTexture( *field, max, -1.0, -slit, -1.0, -slit*field_ratio);
        window_str <<" nz / "<<max<<"\t";
    }

    { //draw potential
    field = &solver.getField( TL_POTENTIAL); 
    max = abs_max(*field);
    //lower right
    drawTexture( *field, max, slit, 1.0, -1.0, -slit*field_ratio);
    window_str <<" phi / "<<max<<"\t";
    }
        
}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    Blueprint bp_mod;
    if( argc == 1)
    {
        bp_mod = read("input.txt");
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
    
    bp.display(cout);
    //construct solvers 
    DFT_DFT_Solver<2> solver2( bp);
    DFT_DFT_Solver<3> solver3( bp);
    if( bp.boundary().bc_x == TL_PERIODIC)
        bp_mod.boundary().bc_x = TL_DST10;
    DRT_DFT_Solver<2> drt_solver2( bp_mod);
    DRT_DFT_Solver<3> drt_solver3( bp_mod);

    const Algorithmic& alg = bp.algorithmic();
    Matrix<double, TL_DFT> ne{ alg.ny, alg.nx, 0.}, nz{ ne}, phi{ ne};
    // place some gaussian blobs in the field
    try{
        //init_gaussian( ne, 0.1,0.2, 10./128./field_ratio, 10./128., amp);
        //init_gaussian( ne, 0.1,0.4, 10./128./field_ratio, 10./128., -amp);
        init_gaussian( ne, 0.8,0.4, 10./128./field_ratio, 10./128., amp);
        //init_gaussian( ne, 0.1,0.8, 10./128./field_ratio, 10./128., -amp);
        //init_gaussian( ni, 0.1,0.5, 0.05/field_ratio, 0.05, amp);
        if( bp.isEnabled( TL_IMPURITY))
        {
            //init_gaussian( nz, 0.8,0.4, 0.05/field_ratio, 0.05, -imp_amp);
            init_gaussian_column( nz, 0.6, 0.05/field_ratio, imp_amp);
        }
        std::array< Matrix<double, TL_DFT>,2> arr2{{ ne, phi}};
        std::array< Matrix<double, TL_DFT>,3> arr3{{ ne, nz, phi}};
        Matrix<double, TL_DRT_DFT> ne_{ alg.ny, alg.nx, 0.}, nz_{ ne_}, phi_{ ne_};
        init_gaussian( ne_, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        //init_gaussian( ne_, 0.2,0.2, 0.05/field_ratio, 0.05, -amp);
        //init_gaussian( ne_, 0.6,0.6, 0.05/field_ratio, 0.05, -amp);
        //init_gaussian( ni_, 0.5,0.5, 0.05/field_ratio, 0.05, amp);
        if( bp.isEnabled( TL_IMPURITY))
        {
            init_gaussian( nz_, 0.5,0.5, 0.05/field_ratio, 0.05, -imp_amp);
            //init_gaussian( nz_, 0.2,0.2, 0.05/field_ratio, 0.05, -imp_amp);
            //init_gaussian( nz_, 0.6,0.6, 0.05/field_ratio, 0.05, -imp_amp);
        }
        std::array< Matrix<double, TL_DRT_DFT>,2> arr2_{{ ne_, phi_}};
        std::array< Matrix<double, TL_DRT_DFT>,3> arr3_{{ ne_, nz_, phi_}};
        //now set the field to be computed
        if( !bp.isEnabled( TL_IMPURITY))
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                solver2.init( arr2, TL_IONS);
            else
                drt_solver2.init( arr2_, TL_IONS);
        }
        else
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                solver3.init( arr3, TL_IONS);
            else
                drt_solver3.init( arr3_, TL_IONS);
        }
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
    while( running)
    {
        overhead.tic();
        //ask if simulation shall be stopped
        glfwPollEvents();
        if( glfwGetKey( 'S')) 
        {
            do
            {
                glfwWaitEvents();
            } while( !glfwGetKey('R') && 
                     !glfwGetKey( GLFW_KEY_ESC) && 
                      glfwGetWindowParam( GLFW_OPENED) );
        }
        
        //draw scene
        if( !bp.isEnabled( TL_IMPURITY))
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                drawScene( solver2);
            else
                drawScene( drt_solver2);
        }
        else
        {
            if( bp.boundary().bc_x == TL_PERIODIC)
                drawScene( solver3);
            else
                drawScene( drt_solver3);
        }
        window_str << setprecision(2) << fixed;
        window_str << " &&   time = "<<t;
        glfwSetWindowTitle( (window_str.str()).c_str() );
        window_str.str("");
        glfwSwapBuffers();
#ifdef TL_DEBUG
        glfwWaitEvents();
        if( glfwGetKey('N'))
        {
#endif
        timer.tic();
        for(unsigned i=0; i<N; i++)
        {
            if( !bp.isEnabled( TL_IMPURITY))
            {
                if( bp.boundary().bc_x == TL_PERIODIC)
                    solver2.step( );
                else
                    drt_solver2.step();
            }
            else
            {
                if( bp.boundary().bc_x == TL_PERIODIC)
                    solver3.step( );
                else
                    drt_solver3.step( );
            }
            t+= alg.dt;
        }
        timer.toc();
#ifdef TL_DEBUG
            cout << "Next "<<N<<" Steps\n";
        }
#endif
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
        overhead.toc();
    }
    glfwTerminate();
    cout << "Average time for one step =                 "<<timer.diff()/(double)N<<"s\n";
    cout << "Overhead for visualisation, etc. per step = "<<(overhead.diff()-timer.diff())/(double)N<<"s\n";
    }
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
    return 0;

}
