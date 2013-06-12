
#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"
#include "nvcc/xspacelib.cuh"
#include "lib/read_input.h"
#include "parameters.h"

#include "hdf5.h"
#include "hdf5_hl.h"

const unsigned n = 3;

int main( int argc, char* argv[])
{
    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile]\n";
        return;
    }
    hid_t file = H5Fopen( argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    H5G_info_t group_info;
    H5Gget_info( file, &group_info);
    hsize_t nlinks = group_info.nlinks;
    std::cout << "Number of groups "<< nlinks<<"\n";
    std::string name; 
    hsize_t length = H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, 0, NULL, 10, H5P_DEFAULT);
    std::cout << "Length of name "<<length<<"\n";
    name.resize( length+1);
    H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, 0, &name[0], length+1, H5P_DEFAULT); //creation order
    std::cout << "Name of first link "<<name<<"\n";
    std::string input;
    
    herr_t  status;
    hsize_t dims[2]; 
    input.resize( 10000);
    status = H5LTread_dataset_string( file, name.data(), &input[0]); //name should precede t so that reading is easier
    const Parameters p( toefl::read_input( input));
    p.display();

    hid_t input_id      = H5Dopen( file, "inputfile" , H5P_DEFAULT);
    hid_t input_space   = H5Dget_space( input_id);
    hssize_t points; 
    points = H5Sget_simple_extent_npoints( input_space );
    H5Sclose( input_space);
    H5Dclose( input_id);
    std::cout << "Size of dataset "<<points<<"\n";




    /*
    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec visual(  grid.size(), 0.);
    dg::DMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    bool running = true;
    double time = 0;
    ab.init( test, y0, p.dt);
    while (running)
    {
        //transform field to an equidistant grid
        if( p.global)
        {
            thrust::transform( y1[0].begin(), y1[0].end(), y1[0].begin(), dg::PLUS<double>(-1));
            dg::blas2::gemv( equi, y1[0], dvisual);
        }
        else
            dg::blas2::gemv( equi, y0[0], dvisual);

        visual = dvisual; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        w.title() << setprecision(2) << scientific;
        w.title() <<"ne / "<<colors.scale()<<"\t";
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);

        //transform phi
        dg::blas2::gemv( test.laplacianM(), test.polarisation(), y1[1]);
        dg::blas2::gemv( equi, y1[1], dvisual);
        visual = dvisual; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        w.title() <<"phi / "<<colors.scale()<<"\t";
        w.title() << fixed; 
        w.title() << " &&   time = "<<time;
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);
#ifdef DG_DEBUG
        glfwWaitEvents();
#endif //DG_DEBUG

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    */
    H5Fclose( file);
    return 0;
}
