#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/xspacelib.cuh"
#include "file/read_input.h"
#include "file/file.h"

#include "galerkin/parameters.h"

const double T = 10;
//is sigma the radius or the diameter

double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}

struct Heaviside2d
{
    Heaviside2d( double sigma):sigma2_(sigma*sigma/4.), x_(0), y_(0){}
    void set_origin( double x0, double y0){ x_=x0, y_=y0;}
    double operator()(double x, double y)
    {
        double r2 = (x-x_)*(x-x_)+(y-y_)*(y-y_);
        if( r2 >= sigma2_)
            return 0.;
        return 1.;
    }
  private:
    const double sigma2_;
    double x_,y_;
};


int main( int argc, char* argv[])
{
    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.h5] \n";
        return -1;
    }
    //std::cout << "Reading: "<< argv[1]<<std::endl;

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    //const unsigned num_out = t5file.get_size();
    int layout = 0;
    if( in.find( "TOEFL") != std::string::npos)
        layout = 0;
    else if( in.find( "INNTO") != std::string::npos)
        layout = 1;
    else 
        std::cerr << "Unknown input file format: default to 0"<<std::endl;
    const Parameters p( file::read_input( in), layout);
    //p.display();
    dg::Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);

    dg::HVec input_h( grid.size());
    dg::HVec input0( input_h), input1(input0), ln0( input0), ln1(input0);
    dg::HVec visual( input0);

    dg::HVec xvec = dg::evaluate( X, grid);
    dg::HVec yvec = dg::evaluate( Y, grid);
    dg::HVec one = dg::evaluate( dg::one, grid);
    dg::HVec w2d = dg::create::w2d( grid);
    dg::HMatrix equi = dg::create::backscatter( grid);

    double posX_max, posY_max;
    //get normalization
    t5file.get_field( input_h, "electrons", 1);
    input0 = input_h;
    if( p.global)
        thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
    Heaviside2d heavi( p.sigma);
    heavi.set_origin( p.lx*p.posX, p.ly*p.posY);
    dg::HVec heavy = dg::evaluate( heavi, grid);
    double normalize = dg::blas2::dot( heavy, w2d, input0);
    double gamma = sqrt( p.kappa*p.n0/(1+p.n0)*(1+p.tau)/p.sigma);//global gamma!!

    unsigned idx = (unsigned)(T/gamma/p.dt/p.itstp) + 1;
    {
        t5file.get_field( input_h, "electrons", idx);
        input0 = input_h;
        if( p.global)
            thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
        //get the maximum amplitude position
        dg::blas2::gemv( equi, input0, visual);
        unsigned position = thrust::distance( visual.begin(), thrust::max_element( visual.begin(), visual.end()) );
        unsigned Nx = p.Nx*p.n; 
        const double hx = grid.hx()/(double)grid.n();
        const double hy = grid.hy()/(double)grid.n();
        posX_max = hx*(1./2. + (double)(position%Nx));
        posY_max = hy*(1./2. + (double)(position/Nx));
        //init Heaviside with this
        heavi.set_origin( posX_max, posY_max);

        dg::HVec heavy = dg::evaluate( heavi, grid);
        std::cout << p.tau<<" "<<p.sigma<<" "<<p.n0<<" ";
        std::cout << std::fixed<< t5file.get_time( idx)*gamma << " ";//(1)
        //std::cout << " "<<posX_max<<" "<<posY_max <<" ";
        std::cout <<std::scientific<< dg::blas2::dot( heavy, w2d, input0)/normalize << "\n";
    }

    return 0;
}

