#include "magnetic_field.h"
#include "solovev.h"
#include "guenther.h"
#include "polynomial.h"
#include "toroidal.h"
//#include "taylor.h" //only if boost is included
#include <dg/file/json_utilities.h>


namespace dg{
namespace geo{

TokamakMagneticField createMagneticField( Json::Value js, file::error mode)
{
    std::string e = file::get( mode, js, "equilibrium", "solovev" ).asString();
    equilibrium equi = str2equilibrium.at( e);
    switch( equi){
        case equilibrium::polynomial:
        {
            polynomial::Parameters gp( js, mode);
            return createPolynomialField( gp);
        }
        case equilibrium::toroidal:
        {
            double R0 = file::get( mode, js, "R_0", 10).asDouble();
            return createToroidalField( R0);
        }
        case equilibrium::guenther:
        {
            double I0 = file::get( mode, js, "I_0", 20).asDouble();
            double R0 = file::get( mode, js, "R_0", 10).asDouble();
            return createGuentherField( R0, I0);
        }
        case equilibrium::circular:
        {
            double I0 = file::get( mode, js, "I_0", 20).asDouble();
            double R0 = file::get( mode, js, "R_0", 10).asDouble();
            return createCircularField( R0, I0);
        }
        default:
        {
            solovev::Parameters gp( js, mode);
            return createSolovevField( gp);
        }
    }
}
TokamakMagneticField createModifiedField( Json::Value js, Json::Value jsmod, file::error mode)
{
    std::string e = file::get( mode, js, "equilibrium", "solovev" ).asString();
    equilibrium equi = str2equilibrium.at( e);
    std::string m = file::get( mode, jsmod, "modifier", "heaviside" ).asString();
    modifier mod = str2modifier.at( m);
    switch (mod) {
        default:
        {
            return createMagneticField( js, mode);
        }
        case modifier::heaviside:
        {
            double psi0 = file::get( mode, jsmod, "psi0", 0. ).asDouble();
            double alpha = file::get( mode, jsmod, "alpha", 0. ).asDouble();
            double sign = file::get( mode, jsmod, "sign", -1. ).asDouble();
            switch( equi){
                case equilibrium::polynomial:
                {
                    polynomial::Parameters gp( js, mode);
                    return createModifiedPolynomialField( gp, psi0, alpha, sign );
                }
                case equilibrium::solovev:
                {
                    solovev::Parameters gp( js, mode);
                    return createModifiedSolovevField( gp, psi0, alpha, sign );
                }
                default:
                    std::stringstream message;
                    message << "*** "<<e<<" has no modification implemented!";
                    if( file::error::is_throw == mode)
                        throw std::runtime_error( message.str());
                    else if (file::error::is_warning == mode)
                        std::cerr <<"WARNING "<< message.str()<<"\n";
                    else
                        ;
                    return createMagneticField( js, mode);
            }
        }
    }
}
} //namespace geo
}//namespace dg
