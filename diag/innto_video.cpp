#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"
#include "file/read_input.h"
#include "file/file.h"

#include "spectral/dft_dft_solver.h"
#include "spectral/blueprint.h"
#include "spectral/particle_density.h"

typedef toefl::DFT_DFT_Solver<2> Sol;
typedef typename Sol::Matrix_Type Mat;

void copyMatrix( const std::vector<double>& src, Mat& dest)
{
    for( unsigned i=0; i<dest.rows(); i++)
        for( unsigned j=0; j<dest.cols(); j++)
            dest(i,j) = src[i*dest.cols()+j];
}
void copyMatrix( const Mat& src, std::vector<double> & dst)
{
    for( unsigned i=0; i<src.rows(); i++)
        for( unsigned j=0; j<src.cols(); j++)
            dst[i*src.cols()+j] = src(i,j);
}

double absmax( double a, double b)
{
    if( fabs(a) > fabs(b)) return fabs(a);
    return fabs(b);
}

int main( int argc, char* argv[])
{
    std::vector<double> v = file::read_input( "innto_window_params.txt");
    draw::HostWindow w(v[3], v[4]);
    w.set_multiplot( v[1], v[2]);

    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile]\n";
        return -1;
    }

    hid_t file = H5Fopen( argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    hsize_t nlinks = file::getNumObjs( file);
    std::string name = file::getName( file, 0);
    std::string in;
    
    herr_t  status;
    hsize_t dims[2]; 
    in.resize( 10000);
    status = H5LTread_dataset_string( file, name.data(), &in[0]); //name should precede t so that reading is easier
    const toefl::Blueprint bp( file::read_input( in));
    bp.display();

    const toefl::Algorithmic alg( bp.algorithmic());
    std::vector<double> visual( alg.nx * alg.ny);
    Mat mat( alg.ny, alg.nx);
    ParticleDensity part( mat, bp);

    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    bool running = true;
    unsigned index = 1;
    std::cout << "PRESS N FOR NEXT FRAME!\n";
    std::cout << "PRESS P FOR PREVIOUS FRAME!\n";
    while (running && index < nlinks )
    {
        hid_t group;
        name = file::getName( file, index);
        //index += v[5];
        group = H5Gopen( file, name.data(), H5P_DEFAULT);

        status = H5LTread_dataset_double(group, "electrons", &visual[0] );
        //compute the color scale
        colors.scale() =  (float)std::accumulate( visual.begin(), visual.end(), 0., absmax);
        //draw electrons 
        w.title() << std::setprecision(2) << std::scientific;
        w.title() <<"ne / "<<colors.scale()<<"\t";
        w.draw( visual, alg.nx, alg.ny, colors);

        status = H5LTread_dataset_double(group, "potential", &visual[0] );
        copyMatrix( visual, mat);
        part.laplace(mat);
        copyMatrix( mat, visual);
        colors.scale() =  (float)std::accumulate( visual.begin(), visual.end(), 0., absmax);
        colors.scale() = 2;
        if( colors.scale() == 0) { colors.scale() = 1;}
        //draw phi and swap buffers
        w.title() <<"omega / "<<colors.scale()<<"\t";
        w.title() << std::fixed; 
        w.title() << " &&  time = "<<file::getTime( name); //read time as double from string
        w.draw( visual, alg.nx, alg.ny, colors);
        bool waiting = true;
        do
        {
            glfwPollEvents();
            if( glfwGetKey( 'B')||glfwGetKey( 'P') ){
                index -= v[5];
                waiting = false;
            }
            else if( glfwGetKey( 'N') ){
                index +=v[5];
                waiting = false;
            }
            //glfwWaitEvents();
        }while( waiting && !glfwGetKey( GLFW_KEY_ESC) && glfwGetWindowParam( GLFW_OPENED));

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    H5Fclose( file);
    return 0;
}
