#pragma once

namespace dg
{


namespace detail{

template<class PsiX, class PsiY, class LaplacePsi, class Chi, class ChiX, class ChiY>
struct LaplaceAdaptPsi
{
    LaplaceAdaptPsi( const PsiX& psiX, const PsiY& psiY, 
    const LaplacePsi& laplacePsi,
    const Chi& chi, const ChiX& chiX, const ChiY& chiY):
        psiX_(psiX), psiY_(psiY), laplacePsi_(laplacePsi),
        chi_(chi), chiX_(chiX), chiY_(chiY){}
    double operator()(double x, double y)
    {
        return psiX_(x,y)*chiX_(x,y)+psiY_(x,y)*chiY_(x,y)+chi_(x,y)*laplacePsi_(x,y);
    }
    private:
    PsiX psiX_;
    PsiY psiY_;
    LaplacePsi laplacePsi_;
    Chi chi_;
    ChiX chiX_;
    ChiY chiY_;
};

template<class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY, class ChiXX, class ChiXY, class ChiYY, class DivChiX, class DivChiY>
struct LaplaceChiPsi
{
    LaplaceChiPsi( const PsiX& psiX, const PsiY& psiY, const PsiXX& psiXX,
    const PsiXY& psiXY, const PsiYY& psiYY, const ChiXX& chiXX, const ChiXY& chiXY,
    const ChiYY& chiYY, const DivChiX& divChiX, const DivChiY& divChiY):
        psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY),
        chiXX_(chiXX), chiXY_(chiXY), chiYY_(chiYY), divChiX_(divChiX), divChiY_(divChiY){}
    double operator()(double x, double y)
    {
        return psiXX_(x,y)*chiXX_(x,y)+2.*psiXY_(x,y)*chiXY_(x,y)+psiYY_(x,y)*chiYY_(x,y)
            + divChiX_(x,y)*psiX_(x,y) + divChiY_(x,y)*psiY_(x,y);
    }

    private:
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
    ChiXX chiXX_;
    ChiXY chiXY_;
    ChiYY chiYY_; 
    DivChiX divChiX_;
    DivChiY divChiY_;
};

template <class PsiXX, class PsiYY>
struct LaplacePsi
{
    LaplacePsi( const PsiXX& psiXX, const PsiYY& psiYY):psiXX_(psiXX), psiYY_(psiYY){}
    double operator()(double x, double y){return psiXX_(x,y)+psiYY_(x,y);}
    private:
        PsiXX psiXX_;
        PsiYY psiYY_;
};

}//namespace detail

template<class PsiX, class PsiY>
struct NablaPsiInv
{
    NablaPsiInv( const PsiX& psiX, const PsiY& psiY):psiX_(psiX), psiY_(psiY){}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y);
        return 1./sqrt(psiX*psiX+psiY*psiY);
    }
    private:
    PsiX psiX_;
    PsiY psiY_;
};

template<class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
struct NablaPsiInvX
{
    NablaPsiInvX( const PsiX& psiX, const PsiY& psiY, const PsiXX& psiXX, const PsiXY& psiXY, const PsiYY& psiYY):
    psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY)
    {}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y);
        double psiXX = psiXX_(x,y), psiXY = psiXY_(x,y);
        double psip = sqrt( psiX*psiX+psiY*psiY);
        return -(psiX*psiXX+psiY*psiXY)/psip/psip/psip;
    }
    
    private:
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
};

template<class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
struct NablaPsiInvY
{
    NablaPsiInvY( const PsiX& psiX, const PsiY& psiY, const PsiXX& psiXX, const PsiXY& psiXY, const PsiYY& psiYY):
    psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY)
    {}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y);
        double psiYY = psiYY_(x,y), psiXY = psiXY_(x,y);
        double psip = sqrt( psiX*psiX+psiY*psiY);
        return -(psiX*psiXY+psiY*psiYY)/psip/psip/psip;
    }
    
    private:
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
};


template<class PsiX, class PsiY>
struct Liseikin_XX
{
    Liseikin_XX(PsiX psiX, PsiY psiY, double k, double eps):k_(k), eps_(eps), psiX_(psiX), psiY_(psiY){}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y), k2 = k_*k_;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return sqrtG*(psiY*psiY+k2*psiX*psiX + eps_);
    }

    private:
    double k_, eps_;
    PsiX psiX_;
    PsiY psiY_;
};
template<class PsiX, class PsiY>
struct Liseikin_XY
{
    Liseikin_XY(PsiX psiX, PsiY psiY, double k, double eps):k_(k), eps_(eps), psiX_(psiX), psiY_(psiY){}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y), k2 = k_*k_;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return sqrtG*(-psiX*psiY+k2*psiX*psiY);
    }

    private:
    double k_, eps_;
    PsiX psiX_;
    PsiY psiY_;
};
template<class PsiX, class PsiY>
struct Liseikin_YY
{
    Liseikin_YY(PsiX psiX, PsiY psiY, double k, double eps):k_(k), eps_(eps), psiX_(psiX), psiY_(psiY){}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y), k2 = k_*k_;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return sqrtG*(+psiX*psiX+k2*psiY*psiY);
    }

    private:
    double k_, eps_;
    PsiX psiX_;
    PsiY psiY_;
};
template<class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
struct DivLiseikinX
{
    DivLiseikinX(PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, double k, double eps):k_(k), eps_(eps), psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY){}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y), k2 = k_*k_;
        double psiXX = psiXX_(x,y), psiXY = psiXY_(x,y), psiYY=psiYY_(x,y);
        double psiY2 = psiY*psiY, psiY3=psiY*psiY2, psiY4=psiY2*psiY2, psiY5=psiY4*psiY;
        double psiX2 = psiX*psiX;
        double psip2 = psiX*psiX+psiY*psiY;
        double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
        return (k2-1.)*(k2*psiY5*psiXY-k2*psiX*psiY4*(psiYY-2*psiXX) +
                    psiX*(eps_+2*eps_*k2+2.*k2*psiX2)*psiY2*psiXX + 
                    psiX*(eps_+k2*psiX2)*((eps_+2*psiX2)*(eps_+k2*psiX2)*psiXY) + 
                    psiY3*(eps_*(1.+k2)*psiXY-(eps_+2.*k2*psiX2)*psiXY))/sqrtG/sqrtG/sqrtG;
    }

    private:
    double k_, eps_;
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
};
template<class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY>
struct DivLiseikinY
{
    DivLiseikinY(PsiX psiX, PsiY psiY, PsiXX psiXX, PsiXY psiXY, PsiYY psiYY, double k, double eps):k_(k), eps_(eps), psiX_(psiX), psiY_(psiY), psiXX_(psiXX), psiXY_(psiXY), psiYY_(psiYY){}
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y), k2 = k_*k_;
        double psiXX = psiXX_(x,y), psiXY = psiXY_(x,y), psiYY=psiYY_(x,y);
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

    private:
    double k_, eps_;
    PsiX psiX_;
    PsiY psiY_;
    PsiXX psiXX_;
    PsiXY psiXY_;
    PsiYY psiYY_;
};


}//namespace dg
