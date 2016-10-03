#pragma once

namespace dg
{

struct NablaPsi
{
    double operator()(double x, double y)
    {
        double psiX = psiX_(x,y), psiY = psiY_(x,y);
        return sqrt(psiX*psiX+psiY*psiY;
    }
};

struct NablaPsiX
{
};

struct NablaPsiY
{
};


template<class PsiX, class PsiY, class PsiXX, class PsiXY, class PsiYY, class ChiXX, class ChiXY, class ChiYY, class DivChiX, class DivChiY>
struct LaplaceChiPsi
{
    LaplaceChipsi( const PsiX& psiX, const PsiY& psiY, const PsiXX& psiXX,
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
}//namespace dg
