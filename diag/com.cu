#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/xspacelib.cuh"
#include "file/read_input.h"
#include "file/file.h"

#include "galerkin/parameters.h"

double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}
template< class container>
void log( const container& y, container& target)
{
    thrust::transform( y.begin(), y.end(), target.begin(), dg::LN<double>());
}

int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.h5] [output.dat]\n";
        return -1;
    }
    //open file for com - data
    //std::string outputfile( argv[1]);
    //outputfile.erase( outputfile.end()-2, outputfile.end());
    //outputfile+="com";
    //std::ofstream os( outputfile.c_str());
    std::ofstream os( argv[2]);

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    const unsigned num_out = t5file.get_size();
    const Parameters p( file::read_input( in));
    p.display();
    dg::Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);

    dg::HVec input_h( grid.size());
    dg::DVec input0( input_h), input1(input0), ln0( input0), ln1(input0);
    std::vector<double> mass, energy, diffusion, dissipation;
    t5file.get_xfile( mass, "mass");
    t5file.get_xfile( energy, "energy");
    t5file.get_xfile( diffusion, "diffusion");
    t5file.get_xfile( dissipation, "dissipation");

    dg::DVec xvec = dg::evaluate( X, grid);
    dg::DVec yvec = dg::evaluate( Y, grid);
    dg::DVec one = dg::evaluate( dg::one, grid);
    dg::DVec w2d = dg::create::w2d( grid);

    double mass_, posX, posY, velX, velY;
    double posX_old = 0, posY_old = 0;
    double deltaT = p.dt*p.itstp;
    os << "#Time(1) posX(2) posY(3) velX(4) velY(5) mass(6) diff(7) (m_tot-m_0)/m_0(8) "
       << "Ue(9) Ui(10) Uphi(11) Utot(12) (U_tot-U_0)/U_0(13) diss(14) \n";
    for( unsigned idx=1; idx<=num_out; idx++)
    {
        t5file.get_field( input_h, "electrons", idx);
        input0 = input_h;
        t5file.get_field( input_h, "electrons", idx);
        input1 = input_h;
        log( input0, ln0), log( input1, ln1);
        double Ue = dg::blas2::dot( input0, w2d, ln0);
        double Ui = p.tau*dg::blas2::dot( input1, w2d, ln1);
        double Uphi = energy[(idx-1)*p.itstp] - Ue - Ui;
        if( p.global)
            thrust::transform( input0.begin(), input0.end(), input0.begin(), dg::PLUS<double>(-1));
        mass_ = dg::blas2::dot( one, w2d, input0 ); 

        posX = dg::blas2::dot( xvec, w2d, input0)/mass_ - p.posX*p.lx;
        posY = dg::blas2::dot( yvec, w2d, input0)/mass_ - p.posY*p.ly;
        velX = (posX - posX_old)/deltaT;
        velY = (posY - posY_old)/deltaT;
        posX_old = posX;
        posY_old = posY;
        //output
        os << t5file.get_time( idx);//(1)
        os << " "<<posX << " " << posY << " "<<velX<<" "<<velY;//(2-5)
        os << " "<<mass[(idx-1)*p.itstp] << " "<<diffusion[(idx-1)*p.itstp];//(6,7)
        os << " "<< (mass[(idx-1)*p.itstp]-mass[0])/(mass[0]-grid.lx()*grid.ly());//blob mass is mass[] - Area (8)
        os << " "<<Ue<<" "<<Ui<<" "<<Uphi<<" "<<energy[(idx-1)*p.itstp]; //(9-12)
        os << " "<<(energy[(idx-1)*p.itstp]-energy[0])/energy[0];//(13)
        os << " "<<dissipation[(idx-1)*p.itstp]; //(14)
        os <<"\n";
    }
    os.close();
    return 0;
}

