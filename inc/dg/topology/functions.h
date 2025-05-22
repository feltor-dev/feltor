
#pragma once
/*! @file
 *
 * @brief Some utility functions for the dg::evaluate routines
 */
#include "../backend/config.h" // DG_DEVICE


namespace dg{
///@addtogroup basics
///@{

/// This enum can be used in \c dg::evaluate

///@brief \f$ f(x, ...) = 0\f$
template<class T, class ...Ts>
DG_DEVICE
T zero(T, Ts ...){return T(0);}

///@brief \f$ f(x, ...) = 1\f$
template<class T, class ...Ts>
DG_DEVICE
T one(T, Ts ...){return T(1);}

///@brief \f$ f(x, ...) = 0\f$
struct ZERO
{
    template<class T, class ...Ts>
    DG_DEVICE
    T operator()(T, Ts ...)const{return T(0);}
};

///@brief \f$ f(x,...) = 1\f$
struct ONE
{
    template<class T, class ...Ts>
    DG_DEVICE
    T operator()(T, Ts ...)const{return T(1);}
};

///@brief \f$ f(x,...) = c\f$
struct CONSTANT
{
    /**
     * @brief Construct with a value
     *
     * @param cte the constant value c
     */
    CONSTANT( double cte): m_value(cte){}

    template<class T, class ...Ts>
    DG_DEVICE
    T operator()(T, Ts ...)const{return T(m_value);}
    private:
    double m_value;
};

///@brief \f$ f(x) = x\f$
DG_DEVICE inline double cooX1d( double x) {return x;}
///@brief \f$ f(x,y) = x\f$
DG_DEVICE inline double cooX2d( double x, double) {return x;}
///@brief \f$ f(x,y,z) = x\f$
DG_DEVICE inline double cooX3d( double x, double, double) {return x;}

///@brief \f$ f(x,y) = y\f$
DG_DEVICE inline double cooY2d( double, double y) {return y;}
///@brief \f$ f(x,y,z) = y\f$
DG_DEVICE inline double cooY3d( double, double y, double) {return y;}
///@brief \f$ f(x,y,z) = z\f$
DG_DEVICE inline double cooZ3d( double, double, double z) {return z;}


///@brief \f$ x = R\sin(\varphi)\f$
DG_DEVICE inline double cooRZP2X( double R, double, double P){ return R*sin(P);}
///@brief \f$ y = R\cos(\varphi)\f$
DG_DEVICE inline double cooRZP2Y( double R, double, double P){ return R*cos(P);}
///@brief \f$ z = Z\f$
DG_DEVICE inline double cooRZP2Z( double, double Z, double){ return Z;}


///@}
} //namespace dg
