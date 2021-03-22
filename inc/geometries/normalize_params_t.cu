#include <iostream>
#include <fstream>
#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "make_field.h"

int main( int argc, char* argv[])
{
    Json::Value js;
    if( argc == 3)
    {
        std::cout << argv[0]<< " "<<argv[1]<<" -> " <<argv[2]<<std::endl;
        dg::file::file2Json( argv[1], js, dg::file::comments::are_discarded);
    }
    else
    {
        std::cerr << "This program reads solovev parameters from an input json file and modifies c[0] such that the resulting Psi_p is zero on the X-point. The resulting parameters are written into an output file, which may overwrite the input file. The program aborts if it is unable to find an X-point\n";
        std::cerr << " Usage: "<< argv[0]<<" [input.json] [normalized.json]\n";
        return -1;
    }

    std::cout << "Input file: \n"<< js.toStyledString();
    dg::file::WrappedJsonValue geom_js( js, dg::file::error::is_warning);
    dg::geo::TokamakMagneticField mag;
    std::string e = geom_js.get( "equilibrium", "solovev" ).asString();
    dg::geo::equilibrium equi = dg::geo::str2equilibrium.at( e);
    switch( equi){
        case dg::geo::equilibrium::polynomial:
        {
            std::cout << "Creating polynomial Field!\n";
            dg::geo::polynomial::Parameters gp( geom_js);
            mag = dg::geo::createPolynomialField( gp);
            break;
        }
        case dg::geo::equilibrium::solovev:
        {
            std::cout << "Creating Solovev Field!\n";
            dg::geo::solovev::Parameters gp( geom_js);
            mag = dg::geo::createSolovevField( gp);
            break;
        }
        default:
        {
            std::cerr << "Equilibrium "<<e<<" cannot be normalized\n";
            return -1;
        }
    }
    double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
    double ZX = -1.1*mag.params().elongation()*mag.params().a();
    try{
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
    }catch ( std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1.;
    }
    const double psipX = mag.psip()( RX, ZX);
    std::cout << "X-point found at "<<RX<<" "<<ZX<<" with Psip = "<<psipX<<std::endl;

    Json::Value output;
    if( equi == dg::geo::equilibrium::solovev)
    {
        dg::geo::solovev::Parameters gp(geom_js);
        gp.c[0] = gp.c[0] - psipX/gp.pp/gp.R_0;
        output = gp.dump();
    }
    else if( equi == dg::geo::equilibrium::polynomial)
    {
        dg::geo::polynomial::Parameters gp(geom_js);
        gp.c[0] = gp.c[0] - psipX/gp.pp/gp.R_0;
        output = gp.dump();
    }
    std::cout << "Output file "<<argv[2]<<": \n"<< output.toStyledString();
    std::fstream file( argv[2], std::fstream::out | std::fstream::trunc);
    file << output.toStyledString();
    file << std::endl;
    file.close();
    return 0;
}

