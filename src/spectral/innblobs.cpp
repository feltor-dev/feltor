#include <iostream>
#include <iomanip>
#include <sstream>
#include <omp.h>

#include "toefl/toefl.h"
#include "file/read_input.h"
#include "draw/host_window.h"
#include "particle_density.h"
#include "dft_dft_solver.h"
//#include "drt_dft_solver.h"
#include "blueprint.h"

/*
 * Reads parameters from given input file (default blobs.in)
 * Inititalizes dft_dft_solver (only periodic BC possible!)
 * visualizes results directly on the screen
 * (difference to innto.cpp lies in the initial condition (blob))
 */

using namespace std;
using namespace toefl;

const unsigned n = 3;
typedef DFT_DFT_Solver<n> Sol;
typedef typename Sol::Matrix_Type Mat;
    
unsigned N; //initialized by init function
double amp, imp_amp; //
double blob_width, posX, posY;
const double slit = 1./500.; //half distance between pictures in units of width
double field_ratio;
unsigned width = 1000, height = 800; //initial window width (height will be computed)
stringstream window_str;  //window name
std::vector<double> visual;
draw::ColorMapRedBlueExt map;

void WindowResize( GLFWwindow* win, int w, int h)
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
    posX = para[23];
    posY = para[24];
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    return bp;
}

    
// The solver has to have the getField( target) function returing M
// and the blueprint() function
template<class Solver>
void drawScene( const Solver& solver, target t, draw::RenderHostData& rend)
{
    ParticleDensity particle( solver.getField( TL_IMPURITIES), solver.blueprint());
    double max;
    const typename Solver::Matrix_Type * field;

    if( t == TL_ALL)
    {
        rend.set_multiplot(2,2);
        { //draw electrons
        field = &solver.getField( TL_ELECTRONS);
        visual = field->copy(); 
        map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
        rend.renderQuad( visual, field->cols(), field->rows(), map);
        window_str << scientific;
        window_str <<"ne / "<<map.scale()<<"\t";
        }

        { //draw Ions
        typename Solver::Matrix_Type ions = solver.getField( TL_IONS);
        particle.linear( ions, solver.getField( TL_POTENTIAL), ions, 0 );
        visual = ions.copy();
        rend.renderQuad( visual, field->cols(), field->rows(), map);
        window_str <<" ni / "<<map.scale()<<"\t";
        }

        if( solver.blueprint().isEnabled( TL_IMPURITY))
        {
            typename Solver::Matrix_Type impurities = solver.getField( TL_IMPURITIES);
            particle.linear( impurities, solver.getField(TL_POTENTIAL), impurities, 0 );
            visual = impurities.copy(); 
            map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
            rend.renderQuad( visual, field->cols(), field->rows(), map);
            window_str <<" nz / "<<max<<"\t";
        }
        else
            rend.renderEmptyQuad();

        { //draw potential
        typename Solver::Matrix_Type phi = solver.getField( TL_POTENTIAL);
        particle.laplace( phi );
        visual = phi.copy(); 
        map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
        rend.renderQuad( visual, field->cols(), field->rows(), map);
        window_str <<" phi / "<<max<<"\t";
        }
    }
    else
    {
        rend.set_multiplot(1,1);
        field = &solver.getField( t);
        visual = field->copy(); 
        map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
        rend.renderQuad( visual, field->cols(), field->rows(), map);
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
        init_gaussian( ne, posX, posY, blob_width/bp.boundary().lx, blob_width/bp.boundary().ly, amp);
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

    height = width/field_ratio;
    GLFWwindow* w = draw::glfwInitAndCreateWindow( width, height, "");
    draw::RenderHostData render( 2,2);
    glfwSetWindowSizeCallback(w, WindowResize);

    glfwSetInputMode( w, GLFW_STICKY_KEYS, GL_TRUE);

    double t = 3*alg.dt;
    Timer timer;
    Timer overhead;
    cout<< "HIT ESC to terminate program \n"
        << "HIT S   to stop simulation \n"
        << "HIT R   to continue simulation!\n";
    target targ = TL_ALL;
    while( !glfwWindowShouldClose(w))
    {
        overhead.tic();
        //ask if simulation shall be stopped
        glfwPollEvents();
        if( glfwGetKey(w, 'S')/*||((unsigned)t%100 == 0)*/) 
        {
            do
            {
                glfwWaitEvents();
            } while( !glfwGetKey(w, 'R') && 
                     !glfwGetKey(w, GLFW_KEY_ESCAPE));
        }
        
        //draw scene
        if( glfwGetKey(w, '1')) targ = TL_ELECTRONS;
        else if( glfwGetKey(w, '2')) targ = TL_IONS;
        else if( glfwGetKey(w, '3')) targ = TL_IMPURITIES;
        else if( glfwGetKey(w, '4')) targ = TL_POTENTIAL;
        else if( glfwGetKey(w, '0')) targ = TL_ALL;
        drawScene(solver, targ, render);
        window_str << setprecision(2) << fixed;
        window_str << " &&   time = "<<t;
        glfwSetWindowTitle(w, (window_str.str()).c_str() );
        window_str.str("");
        glfwSwapBuffers(w);
        timer.tic();
        for(unsigned i=0; i<N; i++)
        {
            solver.step();
            t+= alg.dt;
        }
        timer.toc();
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
