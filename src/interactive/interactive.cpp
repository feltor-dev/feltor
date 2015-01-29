#include <iostream>
#include <iomanip>
#include <sstream>
#include <omp.h>

#include "opencv2/opencv.hpp"

#include "toefl/toefl.h"
#include "file/read_input.h"

#include "draw/host_window.h"
#include "dft_dft_solver.h"
#include "blueprint.h"

/*
 * Reads parameters from given input file
 * Inititalizes the correct solver 
 * visualizes results directly on the screen
 */
    
unsigned N; //initialized by init function
double amp, imp_amp; //
double blob_width, posX, posY;
const double slit = 2./500.; //half distance between pictures in units of width
double field_ratio;
unsigned width = 960, height = 1080; //initial window width & height
std::stringstream window_str;  //window name
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

toefl::Blueprint read( char const * file)
{
    std::cout << "Reading from "<<file<<"\n";
    std::vector<double> para;
    try{ para = file::read_input( file); }
    catch (toefl::Message& m) 
    {  
        m.display(); 
        throw m;
    }
    toefl::Blueprint bp( para);
    amp = para[10];
    imp_amp = para[14];
    N = para[19];
    field_ratio = bp.boundary().lx/bp.boundary().ly;
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
void drawScene( const Solver& solver, draw::RenderHostData& rend)
{
    const typename Solver::Matrix_Type * field;
    
    { //draw electrons
    field = &solver.getField( toefl::TL_ELECTRONS);
    visual = field->copy(); 
    map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
    rend.renderQuad( visual, field->cols(), field->rows(), map);
    window_str << std::scientific;
    window_str <<"ne / "<<map.scale()<<"\t";
    }

    { //draw Ions
    field = &solver.getField( toefl::TL_IONS);
    visual = field->copy();
    //upper right
    rend.renderQuad( visual, field->cols(), field->rows(), map);
    window_str <<" ni / "<<map.scale()<<"\t";
    }

    if( solver.blueprint().isEnabled( toefl::TL_IMPURITY))
    {
        field = &solver.getField( toefl::TL_IMPURITIES); 
        visual = field->copy();
        map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
        //lower left
        rend.renderQuad( visual, field->cols(), field->rows(), map);
        window_str <<" nz / "<<map.scale()<<"\t";
    }
    else
        rend.renderEmptyQuad( );

    { //draw potential
    field = &solver.getField( toefl::TL_POTENTIAL); 
    visual = field->copy(); 
    map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
    rend.renderQuad( visual, field->cols(), field->rows(), map);
    window_str <<" phi / "<<map.scale()<<"\t";
    }
        
}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    toefl::Blueprint bp_mod;
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
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const toefl::Blueprint bp = bp_mod;
    if( bp.boundary().bc_x != toefl::TL_PERIODIC)
    {
        std::cerr << "Only periodic boundaries allowed!\n";
        return -1;
    }
    
    bp.display(std::cout);
    //construct solvers 
    toefl::DFT_DFT_Solver<2> solver2( bp);
    toefl::DFT_DFT_Solver<3> solver3( bp);

    const toefl::Algorithmic& alg = bp.algorithmic();
    const toefl::Boundary& bound = bp.boundary();
    // place some gaussian blobs in the field
    try{
        toefl::Matrix<double, toefl::TL_DFT> ne{ alg.ny, alg.nx, 0.}, nz{ ne}, phi{ ne};
        init_gaussian( ne, posX, posY, blob_width/bound.lx, blob_width/bound.ly, amp);
        if( bp.isEnabled( toefl::TL_IMPURITY))
        {
            //init_gaussian( nz, 0.8,0.4, 0.05/field_ratio, 0.05, -imp_amp);
            init_gaussian_column( nz, 0.6, 0.05/field_ratio, imp_amp);
        }
        std::array< toefl::Matrix<double, toefl::TL_DFT>,2> arr2{{ ne, phi}};
        std::array< toefl::Matrix<double, toefl::TL_DFT>,3> arr3{{ ne, nz, phi}};
        //now set the field to be computed
        if( !bp.isEnabled( toefl::TL_IMPURITY))
        {
            solver2.init( arr2, toefl::TL_IONS);
        }
        else
        {
            solver3.init( arr3, toefl::TL_IONS);
        }
    }catch( toefl::Message& m){m.display();}

    ////////////////////////////////glfw and opencv//////////////////////////////
    cv::VideoCapture cap(0);
    if( !cap.isOpened())
    {
        std::cerr << "Camera not found\n";
        return -1;
    }
    cv::namedWindow( "Camera", cv::WINDOW_NORMAL);
    //draw opencv frame
    cv::Mat frame;
    {

    height = width/field_ratio;
    GLFWwindow* w = draw::glfwInitAndCreateWindow( width, height, "");
    draw::RenderHostData render( 2,2);

    glfwSetWindowSizeCallback( w, WindowResize);
    glfwSetInputMode(w, GLFW_STICKY_KEYS, GL_TRUE);
    //for(;;)
    //{
    //        if(cv::waitKey(30) >= 0) break;
    //}

    double t = 0.;
    toefl::Timer timer;
    toefl::Timer overhead;
    std::cout<< "HIT ESC to terminate program \n"
        << "HIT S   to stop simulation \n"
        << "HIT R   to continue simulation!\n";
    if( !bp.isEnabled( toefl::TL_IMPURITY))
    {
        solver2.first_step();
        solver2.second_step();
    }
    else
    {
        solver3.first_step();
        solver3.second_step();
    }
    t+= 2*alg.dt;
    while( !glfwWindowShouldClose(w))
    {
        cap >> frame; // get a new frame from camera
        overhead.tic();
        //ask if simulation shall be stopped
        glfwPollEvents();
        if( glfwGetKey( w, 'S')) 
        {
            do
            {
                glfwWaitEvents();
            } while( !glfwGetKey(w,  'R') && 
                     !glfwGetKey(w,  GLFW_KEY_ESCAPE));
        }
        
        //draw scene
        if( !bp.isEnabled( toefl::TL_IMPURITY))
        {
            drawScene( solver2, render);
        }
        else
        {
            drawScene( solver3, render);
        }
        window_str << std::setprecision(2) << std::fixed;
        window_str << " &&   time = "<<t;
        glfwSetWindowTitle(w, (window_str.str()).c_str() );
        window_str.str("");
        glfwSwapBuffers( w );
#ifdef TL_DEBUG
        glfwWaitEvents();
        if( glfwGetKey( w,'N'))
        {
#endif
        timer.tic();
        for(unsigned i=0; i<N; i++)
        {
            if( !bp.isEnabled( toefl::TL_IMPURITY))
            {
                toefl::Matrix<double, toefl::TL_DFT> voidmatrix( 2,2,(bool)toefl::TL_VOID);
                solver2.step(voidmatrix);
            }
            else
            {
                toefl::Matrix<double, toefl::TL_DFT> voidmatrix( 2,2,(bool)toefl::TL_VOID);
                solver3.step(voidmatrix );
            }
            t+= alg.dt;
        }
        timer.toc();
#ifdef TL_DEBUG
        std::cout << "Next "<<N<<" Steps\n";
        }
#endif
        overhead.toc();
    }
    glfwTerminate();
    std::cout << "Average time for one step =                 "<<timer.diff()/(double)N<<"s\n";
    std::cout << "Overhead for visualisation, etc. per step = "<<(overhead.diff()-timer.diff())/(double)N<<"s\n";
    }
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
    return 0;

}
