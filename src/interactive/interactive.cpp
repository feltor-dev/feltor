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

    
// The solver has to have the getField( target) function returing M
// and the blueprint() function
template<class Solver>
void drawScene( const Solver& solver, draw::RenderHostData& rend)
{
    const typename Solver::Matrix_Type * field;
    
    { //draw electrons
    field = &solver.getField( toefl::ELECTRONS);
    visual = field->copy(); 
    map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
    rend.renderQuad( visual, field->cols(), field->rows(), map);
    window_str << std::scientific;
    window_str <<"ne / "<<map.scale()<<"\t";
    }

    { //draw Ions
    field = &solver.getField( toefl::IONS);
    visual = field->copy();
    //upper right
    rend.renderQuad( visual, field->cols(), field->rows(), map);
    window_str <<" ni / "<<map.scale()<<"\t";
    }

    if( solver.blueprint().imp)
    {
        field = &solver.getField( toefl::IMPURITIES); 
        visual = field->copy();
        map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
        //lower left
        rend.renderQuad( visual, field->cols(), field->rows(), map);
        window_str <<" nz / "<<map.scale()<<"\t";
    }
    else
        rend.renderEmptyQuad( );

    { //draw potential
    field = &solver.getField( toefl::POTENTIAL); 
    visual = field->copy(); 
    map.scale() = fabs(*std::max_element(visual.begin(), visual.end()));
    rend.renderQuad( visual, field->cols(), field->rows(), map);
    window_str <<" phi / "<<map.scale()<<"\t";
    }
        
}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> para;
    if( argc == 1)
    {
        std::cout << "Reading from input.txt\n";
        try{ para = file::read_input( "input.txt"); }
        catch (toefl::Message& m) 
        {  
            m.display(); 
            throw m;
        }
    }
    else if( argc == 2)
    {
        std::cout << "Reading from "<<argv[1]<<"\n";
        try{ para = file::read_input( argv[1]); }
        catch (toefl::Message& m) 
        {  
            m.display(); 
            throw m;
        }
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    omp_set_num_threads( para[20]);
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    const toefl::Parameters p(para);
    field_ratio = p.lx/p.ly;
    if( p.bc_x != toefl::TL_PERIODIC)
    {
        std::cerr << "Only periodic boundaries allowed!\n";
        return -1;
    }
    
    try{p.consistencyCheck();}
    catch( toefl::Message& m){m.display();throw m;}
    p.display(std::cout);
    //construct solvers 
    toefl::DFT_DFT_Solver<3> solver( p);

    // place some gaussian blobs in the field
    try{
        toefl::Matrix<double, toefl::TL_DFT> ne{ p.ny, p.nx, 0.}, nz{ ne}, phi{ ne};
        init_gaussian( ne, p.posX, p.posY, p.blob_width/p.lx, p.blob_width/p.ly, p.amp);
        init_gaussian_column( nz, 0.6, 0.05/field_ratio, p.imp_amp);
        std::array< toefl::Matrix<double, toefl::TL_DFT>,3> arr3{{ ne, nz, phi}};
        //now set the field to be computed
        solver.init( arr3, toefl::IONS);
    }catch( toefl::Message& m){m.display();}

    cv::VideoCapture cap(0);
    if( !cap.isOpened())
    {
        std::cerr << "Camera not found\n";
        return -1;
    }
    cap.set( CV_CAP_PROP_FRAME_WIDTH,  p.nx);
    cap.set( CV_CAP_PROP_FRAME_HEIGHT, p.ny);
    cv::Mat last, current, flow, vel(p.ny, p.nx, CV_32F);
    std::vector<cv::Mat> v;

    cv::namedWindow("Current",cv::WINDOW_NORMAL);
    cv::namedWindow("Velocity",cv::WINDOW_NORMAL);
    double t = 0.;
    toefl::Timer timer;
    toefl::Timer overhead;
    solver.first_step();
    solver.second_step();
    t+= 2*p.dt;
    toefl::Matrix<double, toefl::TL_DFT> src( p.ny, p.nx, 0.);
    cv::Mat grey, colored, show( p.ny, p.nx, CV_32F);
    cap >> last;
    cv::cvtColor( last, last, CV_BGR2GRAY); //convert colors

    while( true)
    {
        cap >> current; // get a new frame from camera
        cv::cvtColor(current, current, CV_BGR2GRAY); //convert colors
        cv::GaussianBlur(current, current, cv::Size(21,21), 0, 0); //Kernel size, sigma_x, sigma_y
        calcOpticalFlowFarneback(last, current, flow, 0.5, 1, 5, 3,  5, 1.2, 0);
        cv::split( flow, v);
        //erster index y, zweiter index x
        for( unsigned i=0; i<v[0].rows; i++)
            for( unsigned j=0; j<v[0].cols; j++)
                vel.at<float>( i,j) = sqrt( v[0].at<float>(i,j)*v[0].at<float>(i,j) + v[1].at<float>(i,j)*v[1].at<float>(i,j) );
        for( unsigned i=0; i<vel.rows; i++)
            for( unsigned j=0; j<vel.cols; j++)
                if( vel.at<float>(i,j) < 1) vel.at<float>(i,j) = 0;
        //scale velocity to 1 in order to account for distance from camera
        double min, max;
        cv::minMaxLoc( vel, &min, &max);
        std::cout << min <<" "<<max<<std::endl;
        if( max > 1) // if someone is there
            for( unsigned i=0; i<vel.rows; i++)
                for( unsigned j=0; j<vel.cols; j++)
                    vel.at<float>( i,j) /= max;
        cv::flip( vel, vel, +1);
        for( unsigned i=0; i<src.rows(); i++)
            for( unsigned j=0; j<src.cols(); j++)
                src(i,j) = 10*vel.at<double>(i,j);
        overhead.tic();
        const toefl::Matrix<double, toefl::TL_DFT>& field = solver.getField( toefl::IMPURITIES); 
        for( unsigned i=0; i<p.ny; i++)
            for( unsigned j=0; j<p.nx; j++)
                show.at<float>(i,j) = (float)field(i,j);
        cv::minMaxLoc( show, &min, &max);
        show.convertTo(grey, CV_8U, 255.0/(2.*max), 255.0/2.);
        cv::minMaxLoc( grey, &min, &max);

        //cv::applyColorMap( grey, colored, cv::COLORMAP_BONE);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_COOL);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_HOT);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_HSV);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_JET);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_OCEAN); 
        //cv::applyColorMap( grey, colored, cv::COLORMAP_PINK);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_RAINBOW);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_SPRING);
        //cv::applyColorMap( grey, colored, cv::COLORMAP_SUMMER);
        cv::applyColorMap( grey, colored, cv::COLORMAP_AUTUMN);
        cv::applyColorMap( grey, colored, cv::COLORMAP_WINTER);
        window_str << std::setprecision(2) << std::fixed;
        window_str << "time = "<<t;
        //cv::addText( colored, window_str.str(), cv::Point(50,50));
        window_str.str(""); 
        std::cout << colored.rows << " " << colored.cols<<"\n";
        std::cout << vel.rows << " " << vel.cols<<"\n";
        cv::imshow("Current", colored);
        cv::imshow("Velocity", vel);

        timer.tic();
        for(unsigned i=0; i<p.itstp; i++)
        {
            toefl::Matrix<double, toefl::TL_DFT> voidmatrix( 2,2,(bool)toefl::TL_VOID);
            solver.step(src );
            t+= p.dt;
        }
        timer.toc();
        overhead.toc();

        //swap fields
        cv::Mat temp = last;
        last = current;
        current = temp;
        if(cv::waitKey(30) >= 0) break;
    }
    ////////////////////////////////glfw and opencv//////////////////////////////
    std::cout << "Average time for one step =                 "<<timer.diff()/(double)p.itstp<<"s\n";
    std::cout << "Overhead for visualisation, etc. per step = "<<(overhead.diff()-timer.diff())/(double)p.itstp<<"s\n";
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
    return 0;

}
