#include <iostream>
#include <fstream>
#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "solovev.h"
#include "fluxfunctions.h"

int main( int argc, char* argv[])
{
    Json::Value geom_js;
    if( argc == 3)
    {
        std::cout << argv[0]<< " "<<argv[1]<<" -> " <<argv[2]<<std::endl;
        file::file2Json( argv[1], geom_js, file::comments::are_discarded);
    }
    else
    {
        std::cerr << "This program reads solovev parameters from an input json file and subtracts a constant such that the resulting Psi_p is zero on the X-point. The resulting parameters are written into an output file, which may overwrite the input file. This assumes that there is an X-point!\n";
        std::cerr << " Usage: "<< argv[0]<<" [input.json] [normalized.json]\n";
        return -1;
    }
    dg::geo::solovev::Parameters gp(geom_js);
    std::cout << "Input file: \n"<< geom_js.toStyledString();
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    double RX = gp.R_0-1.1*gp.triangularity*gp.a;
    double ZX = -1.1*gp.elongation*gp.a;
    if( gp.hasXpoint())
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
    else
    {
        std::cerr << "Parameters have no X-point!\n";
        return -1;
    }
    const double psipX = mag.psip()( RX, ZX);
    std::cout << "X-point found at "<<RX<<" "<<ZX<<" with Psip = "<<psipX<<std::endl;
    gp.c[0] = gp.c[0] - psipX/gp.pp/gp.R_0;
    Json::Value output = gp.dump();
    std::cout << "Output file "<<argv[2]<<": \n"<< output.toStyledString();
    std::fstream file( argv[2], std::fstream::out | std::fstream::trunc);
    file << output.toStyledString();
    file << std::endl;
    file.close();
    return 0;
}

