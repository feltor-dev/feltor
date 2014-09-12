#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/backend/xspacelib.cuh"
#include "dg/arakawa.h"
#include "dg/functors.h"
#include "file/read_input.h"
#include "file/file.h"
#include "dg/backend/timer.cuh"

#include "toefl/parameters.h"


//read and evaluate TOEFL & INNTO h5 files

double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}
template< class container>
void log( const container& y, container& target)
{
    thrust::transform( y.begin(), y.end(), target.begin(), dg::LN<double>());
}
template <class Matrix, class container>
struct Vesqr
{
    Vesqr( const dg::Grid2d<double>& grid, double kappa): dx( grid.size()), dy(dx), one( grid.size(), 1.), w2d( dg::create::weights(grid)), binv( evaluate( dg::LinearX( kappa, 1.), grid)), arakawa(grid){}
    const container& operator()( const container& phi)
    {
        dg::blas2::gemv( arakawa.dx(), phi, dx);
        dg::blas2::gemv( arakawa.dy(), phi, dy);
        dg::blas1::pointwiseDot( binv, dx, dx);
        dg::blas1::pointwiseDot( binv, dy, dy);
        dg::blas1::pointwiseDot( dx, dx, dx);
        dg::blas1::pointwiseDot( dy, dy, dy);
        dg::blas1::axpby( 1., dx, 1.,  dy);
        return dy;

    }
  private:
    container dx, dy, one, w2d, binv;    
    dg::ArakawaX<Matrix, container> arakawa;

};

int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.h5] [output.dat]\n";
        return -1;
    }
    std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    const unsigned num_out = t5file.get_size();

    int layout = 0;
    if( in.find( "TOEFL") != std::string::npos)
        layout = 0;
    else if( in.find( "INNTO") != std::string::npos)
        layout = 1;
    else 
        std::cerr << "Unknown input file format: default to 0"<<std::endl;
    const Parameters p( file::read_input( in), layout);
    p.display();
    dg::Grid2d<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);

    dg::HVec input_h( grid.size());
    dg::HVec input0( input_h), input1(input0), ln0( input0), ln1(input0);
    dg::HVec visual( input0);
    std::vector<double> mass(p.maxout*p.itstp+1,0), energy(mass), diffusion(mass), dissipation(mass);
    if( p.global)
    {
        t5file.get_xfile( mass, "mass");
        t5file.get_xfile( energy, "energy");
        t5file.get_xfile( diffusion, "diffusion");
        t5file.get_xfile( dissipation, "dissipation");
    }

    dg::HVec xvec = dg::evaluate( X, grid);
    dg::HVec yvec = dg::evaluate( Y, grid);
    dg::HVec one = dg::evaluate( dg::one, grid);
    dg::HVec w2d = dg::create::weights( grid);
    dg::HMatrix equi = dg::create::backscatter( grid);

    t5file.get_field( input0, "electrons", 1);
    if( p.global)
        thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
    double mass_ = dg::blas2::dot( one, w2d, input0 ); 
    const double posX_init = dg::blas2::dot( xvec, w2d, input0)/mass_;
    const double posY_init = dg::blas2::dot( yvec, w2d, input0)/mass_;
    double posX, posY, velX, velY, accX, accY;
    double posX_max, posY_max;
    double posX_old = 0, posY_old = 0;
    double deltaT = p.dt*p.itstp;
    t5file.get_field( input0, "electrons", 2);
        if( p.global)
            thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
        mass_ = dg::blas2::dot( one, w2d, input0 ); 
        posX = dg::blas2::dot( xvec, w2d, input0)/mass_ - posX_init;
        posY = dg::blas2::dot( yvec, w2d, input0)/mass_ - posY_init;
        double velX_old = -posX/deltaT, velY_old = -posY/deltaT; 
        //velX_old = NAN, velY_old = NAN;

    Vesqr<dg::HMatrix, dg::HVec> vesqr( grid, p.kappa);
    os << "#Time(1) posX(2) posY(3) velX(4) velY(5) mass(6) diff(7) (m_tot-m_0)/m_0(8) "
       << "Ue(9) Ui(10) Uphi(11) Utot(12) (U_tot-U_0)/U_0(13) diss(14) posX_max(15) posY_max(16) accX(17) acc(18) max_amp(19)\n";
    //dg::Timer t;
    for( unsigned idx=1; idx<=num_out; idx++)
    {
        //t.tic();
        //std::cout << idx<<std::endl;
        t5file.get_field( input0, "electrons", idx);
        //input0 = input_h;
        t5file.get_field( input1, "ions", idx);
        //input1 = input_h;
        //t.toc();
        //std::cout << "Reading took "<<t.diff()<<"s\n";
        //t.tic();
        double Ue = 0, Ui = 0, Uphi = 0;
        if( p.global)
        {
            //std::cout << "in global branch!\n";
            log( input0, ln0), log( input1, ln1);
            Ue = dg::blas2::dot( input0, w2d, ln0);
            Ui = p.tau*dg::blas2::dot( input1, w2d, ln1);
            Uphi = energy[(idx-1)*p.itstp] - Ue - Ui;
            thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
        }
        mass_ = dg::blas2::dot( one, w2d, input0 ); 
        posX = dg::blas2::dot( xvec, w2d, input0)/mass_ - posX_init;
        posY = dg::blas2::dot( yvec, w2d, input0)/mass_ - posY_init;

        velX = (posX - posX_old)/deltaT;
        velY = (posY - posY_old)/deltaT;
        accX = (velX - velX_old)/deltaT;
        accY = (velY - velY_old)/deltaT;
        posX_old = posX; posY_old = posY;
        velX_old = velX; velY_old = velY;
        //output
        os << t5file.get_time( idx);//(1)
        os << " "<<posX << " " << posY << " "<<velX<<" "<<velY;//(2-5)
        os << " "<<mass[(idx-1)*p.itstp] << " "<<diffusion[(idx-1)*p.itstp];//(6,7)
        os << " "<< (mass[(idx-1)*p.itstp]-mass[0])/(mass[0]-grid.lx()*grid.ly());//blob mass is mass[] - Area (8)
        os << " "<<Ue<<" "<<Ui<<" "<<Uphi<<" "<<energy[(idx-1)*p.itstp]; //(9-12)
        os << " "<<(energy[(idx-1)*p.itstp]-energy[0])/energy[0];//(13)
        os << " "<<dissipation[(idx-1)*p.itstp]; //(14)
        //get the maximum amplitude position
        dg::blas2::gemv( equi, input0, visual);
        unsigned position = thrust::distance( visual.begin(), thrust::max_element( visual.begin(), visual.end()) );
        unsigned Nx = p.Nx*p.n; 
        const double hx = grid.hx()/(double)grid.n();
        const double hy = grid.hy()/(double)grid.n();
        posX_max = hx*(1./2. + (double)(position%Nx))-posX_init;
        posY_max = hy*(1./2. + (double)(position/Nx))-posY_init;
        os << " "<<posX_max<<" "<<posY_max;
        os << " "<<accX<<" "<<accY;
        os << " "<<*thrust::max_element( visual.begin(), visual.end());
        os <<"\n";
        //t.toc();
        //std::cout << "The rest took "<<t.diff()<<"s\n";
    }
    os.close();
    return 0;
}

