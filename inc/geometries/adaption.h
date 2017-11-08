#pragma once

#include "fluxfunctions.h"

/*!@file 
 *
 * contains an adaption function and a monitor metric for the Hector algorithm
 *
 */
namespace dg
{
namespace geo
{

///@cond
namespace detail{

struct LaplaceAdaptPsi: public aCloneableBinaryFunctor<LaplaceAdaptPsi>
{
    LaplaceAdaptPsi( const BinaryFunctorsLvl2& psi, const BinaryFunctorsLvl1& chi) : psi_(psi), chi_(chi){}
    private:
    double do_compute(double x, double y)const
    {
        return  psi_.dfx()(x,y)*chi_.dfx()(x,y) +
                psi_.dfy()(x,y)*chi_.dfy()(x,y) +
                chi_.f()(x,y)*( psi_.dfxx()(x,y) + psi_.dfyy()(x,y));
    }
    BinaryFunctorsLvl2 psi_;
    BinaryFunctorsLvl1 chi_;
};

struct LaplaceChiPsi: public aCloneableBinaryFunctor<LaplaceChiPsi>
{
    LaplaceChiPsi( const BinaryFunctorsLvl2& psi, const BinarySymmTensorLvl1& chi):
        psi_(psi), chi_(chi){}
    private:
    double do_compute(double x, double y)const
    {
        return psi_.dfxx()(x,y)*chi_.xx()(x,y)+2.*psi_.dfxy()(x,y)*chi_.xy()(x,y)+psi_.dfyy()(x,y)*chi_.yy()(x,y)
            + chi_.divX()(x,y)*psi_.dfx()(x,y) + chi_.divY()(x,y)*psi_.dfy()(x,y);
    }

    BinaryFunctorsLvl2 psi_;
    BinarySymmTensorLvl1 chi_;
};

struct LaplacePsi: public aCloneableBinaryFunctor<LaplacePsi>
{
    LaplacePsi( const BinaryFunctorsLvl2& psi): psi_(psi){}
    private:
    double do_compute(double x, double y)const{return psi_.dfxx()(x,y)+psi_.dfyy()(x,y);}
    BinaryFunctorsLvl2 psi_;
};

}//namespace detail
///@endcond

///@addtogroup profiles
///@{

/**
 * @brief  A weight function for the Hector algorithm
 *\f[ |\nabla\psi|^{-1} = (\psi_x^2 + \psi_y^2)^{-1/2} \f]
 */
struct NablaPsiInv: public aCloneableBinaryFunctor<NablaPsiInv>
{
    /**
     * @brief Construct with function container
     *
     * @param psi \f$ \psi(x,y)\f$ and its first derivatives
     */
    NablaPsiInv( const BinaryFunctorsLvl1& psi): psi_(psi){}
    private:
    virtual double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y);
        return 1./sqrt(psiX*psiX+psiY*psiY);
    }
    BinaryFunctorsLvl1 psi_;
};

/**
 * @brief Derivative of the weight function
 *\f[\partial_x|\nabla\psi|^{-1} \f]
 */
struct NablaPsiInvX: public aCloneableBinaryFunctor<NablaPsiInvX>
{
    NablaPsiInvX( const BinaryFunctorsLvl2& psi):psi_(psi) {}
    private:
    virtual double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y);
        double psiXX = psi_.dfxx()(x,y), psiXY = psi_.dfxy()(x,y);
        double psip = sqrt( psiX*psiX+psiY*psiY);
        return -(psiX*psiXX+psiY*psiXY)/psip/psip/psip;
    }
    
    BinaryFunctorsLvl2 psi_;
};

/**
 * @brief Derivative of the weight function
 *\f[ \partial_y|\nabla\psi|^{-1} \f]
 */
struct NablaPsiInvY: public aCloneableBinaryFunctor<NablaPsiInvY>
{
    NablaPsiInvY( const BinaryFunctorsLvl2& psi):psi_(psi) {}
    private:
    double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y);
        double psiYY = psi_.dfyy()(x,y), psiXY = psi_.dfxy()(x,y);
        double psip = sqrt( psiX*psiX+psiY*psiY);
        return -(psiX*psiXY+psiY*psiYY)/psip/psip/psip;
    }
    
    BinaryFunctorsLvl2 psi_;
};

/**
 * @brief A container class that contains all NablaPsiInv functors
 */
BinaryFunctorsLvl1 make_NablaPsiInvCollective( const BinaryFunctorsLvl2& psi)
{
    BinaryFunctorsLvl1 temp( new NablaPsiInv(psi), new NablaPsiInvX(psi), new NablaPsiInvY( psi));
    return temp;
}



/**
 * @brief The xx-component of the Liseikin monitor metric
 * \f[ \chi^{xx} = (\psi_y^2+k^2\psi_x^2 + \varepsilon)/\sqrt{\det \chi} \f] 
 *
 * with
 * \f[ \det \chi = (\varepsilon+(\nabla\psi)^2)(\varepsilon+k^2(\nabla\psi)^2)\f]
 */
struct Liseikin_XX: public aCloneableBinaryFunctor<Liseikin_XX>
{
    Liseikin_XX(const BinaryFunctorsLvl1& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
    private:
    double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return (psiY*psiY+k2*psiX*psiX + eps_)/sqrtG;
    }

    double k_, eps_;
    BinaryFunctorsLvl1 psi_;
};

/**
 * @brief The xy-component of the Liseikin monitor metric
 * \f[ \chi^{xy} = (-\psi_x\psi_y+k^2\psi_x\psi_y )/\sqrt{\det \chi} \f] 
 *
 * with
 * \f[ \det \chi = (\varepsilon+(\nabla\psi)^2)(\varepsilon+k^2(\nabla\psi)^2)\f]
 */
struct Liseikin_XY: public aCloneableBinaryFunctor<Liseikin_XY>
{
    Liseikin_XY(const BinaryFunctorsLvl1& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
    private:
    double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return (-psiX*psiY+k2*psiX*psiY)/sqrtG;
    }

    double k_, eps_;
    BinaryFunctorsLvl1 psi_;
};

/**
 * @brief The yy-component of the Liseikin monitor metric
 * \f[ \chi^{yy} = (\varepsilon+\psi_x^2+k^2\psi_y^2 )/\sqrt{\det \chi} \f] 
 *
 * with
 * \f[ \det \chi = (\varepsilon+(\nabla\psi)^2)(\varepsilon+k^2(\nabla\psi)^2)\f]
 */
struct Liseikin_YY: public aCloneableBinaryFunctor<Liseikin_YY>
{
    Liseikin_YY(const BinaryFunctorsLvl1& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
    private:
    double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return (eps_+psiX*psiX+k2*psiY*psiY)/sqrtG;
    }

    double k_, eps_;
    BinaryFunctorsLvl1 psi_;
};

/**
 * @brief The x-component of the divergence of the Liseikin monitor metric
 * \f[ \partial_x \chi^{xx} + \partial_y\chi^{yx}\f]
 */
struct DivLiseikinX: public aCloneableBinaryFunctor<DivLiseikinX>
{
    DivLiseikinX(const BinaryFunctorsLvl2& psi, double k, double eps): k_(k), eps_(eps), psi_(psi){}
    private:
    double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
        double psiXX = psi_.dfxx()(x,y), psiXY = psi_.dfxy()(x,y), psiYY=psi_.dfyy()(x,y);
        double psiY2 = psiY*psiY, psiY3=psiY*psiY2, psiY4=psiY2*psiY2, psiY5=psiY4*psiY;
        double psiX2 = psiX*psiX, psiX4=psiX2*psiX2;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return (k2-1.)*(k2*psiY5*psiXY-k2*psiX*psiY4*(psiYY-2.*psiXX) +
                    psiX*(eps_+2.*eps_*k2+2.*k2*psiX2)*psiY2*psiXX + 
                    psiX*(eps_+k2*psiX2)*((eps_+psiX2)*psiYY+eps_*psiXX)+
                    psiY*((eps_*eps_-k2*psiX4)*psiXY-(eps_+2*psiX2)*(eps_+k2*psiX2)*psiXY) + 
                    psiY3*(eps_*(1.+k2)*psiXY-(eps_+2.*k2*psiX2)*psiXY))/sqrtG/sqrtG/sqrtG;
    }

    double k_, eps_;
    BinaryFunctorsLvl2 psi_;
};

/**
 * @brief The y-component of the divergence of the Liseikin monitor metric
 * \f[ \partial_x \chi^{xy} + \partial_y\chi^{yy}\f]
 */
struct DivLiseikinY : public aCloneableBinaryFunctor<DivLiseikinY>
{
    DivLiseikinY(const BinaryFunctorsLvl2& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
    private:
    double do_compute(double x, double y)const
    {
        double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
        double psiXX = psi_.dfxx()(x,y), psiXY = psi_.dfxy()(x,y), psiYY=psi_.dfyy()(x,y);
        double psiX2 = psiX*psiX, psiX3=psiX*psiX2, psiX4=psiX2*psiX2, psiX5=psiX4*psiX;
        double psiY2 = psiY*psiY, psiY4 = psiY2*psiY2;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return (k2-1.)*(psiX2*psiY*(eps_+2.*eps_*k2+2.*k2*psiY2)*psiYY + 
                k2*psiX4*psiY*(2.*psiYY-psiXX)+psiY*(eps_+k2*psiY2)
                *(eps_*psiYY+(eps_+psiY2)*psiXX)+k2*psiX5*psiXY+
                psiX3*(-(eps_+2.*k2*psiY2)*psiXY+eps_*(1.+k2)*psiXY) +
                psiX*(-(eps_+2.*psiY2)*(eps_+k2*psiY2)*psiXY + (eps_*eps_-k2*psiY4)*psiXY))/sqrtG/sqrtG/sqrtG;
    }

    double k_, eps_;
    BinaryFunctorsLvl2 psi_;
};

BinarySymmTensorLvl1 make_LiseikinCollective( const BinaryFunctorsLvl2& psi, double k, double eps)
{
    BinarySymmTensorLvl1 temp( new Liseikin_XX(psi,k,eps), new Liseikin_XY(psi,k,eps), new Liseikin_YY(psi,k,eps), new DivLiseikinX(psi,k,eps), new DivLiseikinY(psi,k,eps));
    return temp;
}
///@}

}//namespace geo
}//namespace dg
