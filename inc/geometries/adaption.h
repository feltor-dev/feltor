#pragma once

namespace dg
{

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
    psiX_(psiX), psiY_(psiY), psiXX_(psiX), psiXY_(psiXY),
    psiYY_(psiYY)
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
    psiX_(psiX), psiY_(psiY), psiXX_(psiX), psiXY_(psiXY),
    psiYY_(psiYY)
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

}//namespace dg
