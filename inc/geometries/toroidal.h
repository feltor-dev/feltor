#pragma once

#include "magnetic_field.h"

namespace dg{
namespace geo{
namespace toroidal{

struct Constant:public aCloneableBinaryOperator<Constant> 
{ 
    Constant(double c):c_(c){}
    double operator()(double R,double Z)const{return c_;}
    private:
    double c_;
};
/**
 * @brief Models a slab toroidal field 
 *
 * \f$ B=\frac{R_0}{R}\f$, \f$ \psi_p = 1\f$ and \f$ I = 1\f$.
 @note The solovev field can also be made to model a todoidal slab field
 */
struct MagneticField : public dg::geo::aTokamakMagneticField
{
    MagneticField( double R0): aTokamakMagneticField(R0, 
        new Constant(1), 
        new Constant(0), 
        new Constant(0), 
        new Constant(0), 
        new Constant(0), 
        new Constant(0), 
        new Constant(0), 
        new Constant(1), 
        new Constant(0), 
        new Constant(0)){}
};

}//namespace toroidal
}//namespace geo
}//namespace dg
