//Mathematica generated file. Do not manually edit!
//Manufactures a solution for the feltor equations for circular flux-surfaces with R0=10, I0=20, a=1 and all Dirichlet boundary conditions
//Defines the functors: 
//    Ne, Ni, Ue, Ui, Phie, Phii, GammaPhie, GammaNi, A
//as well as
//    SNe, SNi, SUe, SUi, SPhie, SPhii, SGammaPhie, SGammaNi, SA
//Requires a definition for DG_DEVICE (define empty for host code and __host__ __device__ to generate CUDA code!)
//Requires a definition for FELTORPERP and FELTORPARALLEL (define 1/0 to activate/deactive respective terms in equations)
#include <cmath>
namespace feltor{ namespace manufactured{ 
DG_DEVICE double Power(double x, double y){ return pow(x,y);}
DG_DEVICE double Sqrt(double x){ return sqrt(x);}
DG_DEVICE double Sin(double x){ return sin(x);}
DG_DEVICE double Cos(double x){ return cos(x);}
static const double Pi = M_PI;
struct Ne{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct Ni{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct Ue{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/(3.*Sqrt(-mue))
; }};
struct Ui{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3.
; }};
struct Phie{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.
; }};
struct Phii{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.
; }};
struct GammaPhie{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.
; }};
struct GammaNi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct A{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (beta*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/4.
; }};
struct SNe{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1.5707963267948966*Cos(Pi*t)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*Z) + 
  (FELTORPARALLEL*(0. - (2.0943951023931953*(-10 + R)*R*Cos(2*Pi*Z)*
          Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
       (0.5235987755982988*(-10 + R)*R*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
          Sin(Pi*(-10 + R))*Sin(2*Pi*(-10 + R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       (0.5235987755982988*R*Z*Cos(Pi*(-10 + R))*Sin(P)*Sin(2*P)*
          Sin(2*Pi*(-10 + R))*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       (3.3333333333333335*Cos(P)*Sin(2*P)*Sin(Pi*(-10 + R))*
          Sin(2*Pi*(-10 + R))*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       (2.0943951023931953*R*Z*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       (40*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/
        (3.*Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       (0.3333333333333333*Z*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/R + 
  FELTORPERP*(-((nuperp*(0. + 1.5707963267948966*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z) - 
            9.869604401089358*R*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)\
))/R) + (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((0.15000000000000002*R*(-20. + 2.*R)*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.1*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (9*Power(Pi,2)*R*Cos(3*Pi*R)*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (3*Pi*R*(-20. + 2.*R)*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*
              Sin(3*Pi*t))/
            (100.*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)) \
- (3*Pi*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (2*beta*Power(Pi,2)*R*Cos(4*Pi*R)*Cos(4*Pi*Z)*Sin(2*P)*
              Sin(4*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))) + 
           (beta*Power(Pi,2)*R*Cos(2*Pi*(-10 + R))*Cos(4*Pi*Z)*Sin(2*P)*
              Sin(4*P)*Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))) - 
           (beta*Pi*R*(-20. + 2.*R)*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*
              Sin(2*Pi*(-10 + R))*Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z))/
            (60.*Sqrt(-mue)*Power(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2),1.5)) + 
           (beta*Pi*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2)))) + 
        1.5707963267948966*R*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z)*
         ((-0.1*R*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (3*Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (beta*Pi*R*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2)))) + 
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((-0.1*R*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (3*Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (beta*Pi*R*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2)))) + 
        R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((-3.*taue*Z*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.2*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (9*Power(Pi,2)*R*Cos(3*Pi*R)*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (2*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
            (45.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (beta*Pi*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z))/
            (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))) - 
           (0.011111111111111112*Z*
              (-9*taue + Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
                 Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (0.18849555921538758*R*Z*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*t)*
              Sin(3*Pi*Z))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (beta*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(4*Pi*Z))/
            (60.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))) + 
           (0.008333333333333333*beta*Z*Sin(2*P)*Sin(4*P)*
              Sin(2*Pi*(-10 + R))*(4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*
              Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z)*Sin(4*Pi*Z))/
            (Sqrt(-mue)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
               1.5))) + 1.5707963267948966*R*Cos(Pi*Z)*Sin(P)*
         Sin(Pi*(-10 + R))*Sin(Pi*t)*
         ((taue*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (-9*taue + Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
               Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
            (90.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (3*Pi*R*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (beta*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z)*Sin(4*Pi*Z))/
            (120.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2)))))/R)
; }};
struct SNi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1.5707963267948966*Cos(Pi*t)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*Z) + 
  (FELTORPARALLEL*(0. - (2.0943951023931953*(-10 + R)*R*Cos(2*Pi*Z)*
          Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)))/
        Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
       (0.5235987755982988*(-10 + R)*R*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
          Sin(Pi*(-10 + R))*Sin(2*Pi*(-10 + R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(2*Pi*Z))/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (0.5235987755982988*R*Z*Cos(Pi*(-10 + R))*Sin(P)*Sin(2*P)*
          Sin(2*Pi*(-10 + R))*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
        Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (3.3333333333333335*Cos(P)*Sin(2*P)*Sin(Pi*(-10 + R))*
          Sin(2*Pi*(-10 + R))*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
        Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (2.0943951023931953*R*Z*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (40*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/(3.*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) \
+ (0.3333333333333333*Z*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/R + 
  FELTORPERP*(-((nuperp*(0. + 1.5707963267948966*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z) - 
            9.869604401089358*R*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)\
))/R) + (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((0.15000000000000002*R*(-20. + 2.*R)*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.1*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (9*Power(Pi,2)*R*Cos(3*Pi*R)*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (3*Pi*R*(-20. + 2.*R)*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*
              Sin(3*Pi*t))/
            (100.*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)) \
- (3*Pi*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (2*beta*Power(Pi,2)*R*Cos(4*Pi*R)*Cos(4*Pi*Z)*Sin(2*P)*
              Sin(4*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z))/
            (15.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (beta*Power(Pi,2)*R*Cos(2*Pi*(-10 + R))*Cos(4*Pi*Z)*Sin(2*P)*
              Sin(4*P)*Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (15.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (beta*Pi*R*(-20. + 2.*R)*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*
              Sin(2*Pi*(-10 + R))*Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z))/
            (60.*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)) + 
           (beta*Pi*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (30.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
        1.5707963267948966*R*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z)*
         ((-0.1*R*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (3*Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (beta*Pi*R*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (30.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((-0.1*R*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (3*Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (beta*Pi*R*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
            (30.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
        R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((-3.*taui*Z*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.2*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (9*Power(Pi,2)*R*Cos(3*Pi*R)*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*t))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (2*mui*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
            (45.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (beta*Pi*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z))/
            (30.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (0.011111111111111112*Z*
              (-9*taui - mui*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
                 Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (0.18849555921538758*R*Z*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*t)*
              Sin(3*Pi*Z))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (beta*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(4*Pi*Z))/
            (60.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (0.008333333333333333*beta*Z*Sin(2*P)*Sin(4*P)*
              Sin(2*Pi*(-10 + R))*(4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*
              Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z)*Sin(4*Pi*Z))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)) + 
        1.5707963267948966*R*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
         ((taui*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (-9*taui - mui*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
               Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
            (90.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (3*Pi*R*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
            (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (beta*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
              (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
              Sin(2*Pi*Z)*Sin(4*Pi*Z))/
            (120.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))/R)
; }};
struct SUe{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (2*Pi*Cos(2*Pi*t)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*Z))/(3.*Sqrt(-mue)) + 
  FELTORPARALLEL*((taue*Sin(Pi*t)*
        ((15.707963267948966 - 1.5707963267948966*R)*R*Cos(Pi*Z)*Sin(P)*
           Sin(Pi*R) + (1.5707963267948966*R*Z*Cos(Pi*R)*Sin(P) + 
             10.*Cos(P)*Sin(Pi*R))*Sin(Pi*Z)))/
      (mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     (0.6981317007977318*(-10 + R)*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
        Power(Sin(2*Pi*(-10 + R)),2)*Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
      (mue*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
     (0.6981317007977318*Z*Cos(2*Pi*(-10 + R))*Power(Sin(2*P),2)*
        Sin(2*Pi*(-10 + R))*Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
      (mue*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
     (40*Cos(2*P)*Sin(2*P)*Power(Sin(2*Pi*(-10 + R)),2)*
        Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
      (9.*mue*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
     (nuparallele*(0. + (1.*Z*
             ((-2.0943951023931953*(-10 + R)*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) + 
               (2.0943951023931953*Z*Cos(2*Pi*(-10 + R))*Sin(2*P)*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) + 
               (40*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (3.*Sqrt(-mue)*R*
                  Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (1.*(-10 + R)*R*((13.15947253478581*Z*Cos(2*Pi*(-10 + R))*
                  Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) + 
               (80*Pi*Cos(2*P)*Cos(2*Pi*Z)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t))/
                (3.*Sqrt(-mue)*R*
                  Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
               (2.0943951023931953*(-10 + R)*Z*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                (Sqrt(-mue)*Power(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2),1.5)) - 
               (2.0943951023931953*Power(Z,2)*Cos(2*Pi*(-10 + R))*
                  Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*Power(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2),1.5)) + 
               (2.0943951023931953*Cos(2*Pi*(-10 + R))*Sin(2*P)*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) - 
               (13.333333333333334*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*R*
                  Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) \
+ (13.15947253478581*(-10 + R)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20*((-4.1887902047863905*(-10 + R)*Cos(2*P)*Cos(2*Pi*Z)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) + 
               (4.1887902047863905*Z*Cos(2*P)*Cos(2*Pi*(-10 + R))*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) - 
               (80*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (3.*Sqrt(-mue)*R*
                  Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (1.*R*Z*((-13.15947253478581*(-10 + R)*Cos(2*Pi*(-10 + R))*
                  Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) + 
               (2.0943951023931953*Power(-10 + R,2)*Cos(2*Pi*Z)*
                  Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                (Sqrt(-mue)*Power(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2),1.5)) - 
               (2.0943951023931953*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) + 
               (80*Pi*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/
                (3.*Sqrt(-mue)*R*
                  Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
               (2.0943951023931953*(-10 + R)*Z*Cos(2*Pi*(-10 + R))*
                  Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*Power(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2),1.5)) - 
               (13.333333333333334*(-10 + R)*Cos(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*R*
                  Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) - 
               (40*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (3.*Sqrt(-mue)*Power(R,2)*
                  Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
               (13.15947253478581*Z*Sin(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
      (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     (Sin(3*Pi*t)*((18.84955592153876 - 1.8849555921538759*R)*R*
           Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R) + 
          (1.8849555921538759*R*Z*Cos(3*Pi*R)*Sin(3*P) + 
             12.*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*Z)))/
      (mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
  (beta*Pi*Cos(4*Pi*t)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*Z))/mue + 
  FELTORPERP*(0. - (0.03333333333333333*taue*Z*Sin(2*P)*Sin(2*Pi*(-10 + R))*
        Sin(2*Pi*t)*Sin(2*Pi*Z))/
      (Sqrt(-mue)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)) - 
     (Pi*Cos(3*Pi*Z)*Sin(2*P)*Sin(3*P)*Sin(2*Pi*(-10 + R))*Sin(3*Pi*R)*
        Sin(2*Pi*t)*Sin(3*Pi*t)*Sin(2*Pi*Z))/
      (50.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
     (eta*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        ((Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. - 
          (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (3.*Sqrt(-mue))))/mue - 
     (nuperp*((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (3.*Sqrt(-mue)) - (8*Power(Pi,2)*R*Sin(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/(3.*Sqrt(-mue))))/R + 
     (2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        ((-0.1*R*taue*Z)/
           Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
          (3*Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
           (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
          (beta*Pi*R*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
             Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
           (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
               1.*Power(Z,2)))))/(3.*Sqrt(-mue)) + 
     ((-0.20943951023931953*R*taue*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*R)*
           Sin(2*Pi*t)*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
         (Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
        (0.10471975511965977*R*taue*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
           Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
         (Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
        (0.03333333333333333*R*taue*Z*Sin(2*P)*Sin(2*Pi*R)*Sin(2*Pi*t)*
           (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
         (Sqrt(-mue)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
            1.5)))/(R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) \
+ (beta*taue*Sin(P)*Sin(4*P)*Sin(Pi*t)*Sin(4*Pi*t)*
        (0.4934802200544679*R*Cos(Pi*R)*Cos(4*Pi*Z)*Sin(4*Pi*R)*Sin(Pi*Z) + 
          Cos(Pi*Z)*Sin(Pi*R)*(-0.4934802200544679*R*Cos(4*Pi*R) - 
             0.039269908169872414*Sin(4*Pi*R))*Sin(4*Pi*Z)))/
      (mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     (3*beta*Pi*Sin(3*P)*Sin(4*P)*Sin(3*Pi*t)*Sin(4*Pi*t)*
        (4*Pi*R*Cos(3*Pi*R)*Cos(4*Pi*Z)*Sin(4*Pi*R)*Sin(3*Pi*Z) - 
          Cos(3*Pi*Z)*Sin(3*Pi*R)*(4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*
           Sin(4*Pi*Z)))/
      (200.*mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
     (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
        ((taue*(-50. + 1.*R - 0.1*Power(Z,2)))/
           Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
          (-9*taue + Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
           (90.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
          (3*Pi*R*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
          (beta*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
             (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
             Sin(2*Pi*Z)*Sin(4*Pi*Z))/
           (120.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
               1.*Power(Z,2)))))/(3.*Sqrt(-mue)))
; }};
struct SUi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (2*Pi*Cos(2*Pi*t)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*Z))/3. + 
  FELTORPARALLEL*((taui*Sin(Pi*t)*
        ((15.707963267948966 - 1.5707963267948966*R)*R*Cos(Pi*Z)*Sin(P)*
           Sin(Pi*R) + (1.5707963267948966*R*Z*Cos(Pi*R)*Sin(P) + 
             10.*Cos(P)*Sin(Pi*R))*Sin(Pi*Z)))/
      (mui*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) - 
     (0.6981317007977318*(-10 + R)*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
        Power(Sin(2*Pi*(-10 + R)),2)*Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
      Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
     (0.6981317007977318*Z*Cos(2*Pi*(-10 + R))*Power(Sin(2*P),2)*
        Sin(2*Pi*(-10 + R))*Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
      Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
     (40*Cos(2*P)*Sin(2*P)*Power(Sin(2*Pi*(-10 + R)),2)*
        Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
      (9.*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
     (nuparalleli*(0. + (1.*Z*
             ((-2.0943951023931953*(-10 + R)*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (2.0943951023931953*Z*Cos(2*Pi*(-10 + R))*Sin(2*P)*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (40*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (3.*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (1.*(-10 + R)*R*((13.15947253478581*Z*Cos(2*Pi*(-10 + R))*
                  Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (80*Pi*Cos(2*P)*Cos(2*Pi*Z)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t))/
                (3.*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
               (2.0943951023931953*(-10 + R)*Z*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
               (2.0943951023931953*Power(Z,2)*Cos(2*Pi*(-10 + R))*
                  Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) + 
               (2.0943951023931953*Cos(2*Pi*(-10 + R))*Sin(2*P)*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
               (13.333333333333334*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) \
+ (13.15947253478581*(-10 + R)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20*((-4.1887902047863905*(-10 + R)*Cos(2*P)*Cos(2*Pi*Z)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (4.1887902047863905*Z*Cos(2*P)*Cos(2*Pi*(-10 + R))*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
               (80*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (3.*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (1.*R*Z*((-13.15947253478581*(-10 + R)*Cos(2*Pi*(-10 + R))*
                  Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (2.0943951023931953*Power(-10 + R,2)*Cos(2*Pi*Z)*
                  Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
               (2.0943951023931953*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (80*Pi*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/
                (3.*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
               (2.0943951023931953*(-10 + R)*Z*Cos(2*Pi*(-10 + R))*
                  Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
               (13.333333333333334*(-10 + R)*Cos(2*P)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) \
- (40*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                (3.*Power(R,2)*
                  Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
               (13.15947253478581*Z*Sin(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t)*Sin(2*Pi*Z))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
      (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     (Sin(3*Pi*t)*((18.84955592153876 - 1.8849555921538759*R)*R*
           Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R) + 
          (1.8849555921538759*R*Z*Cos(3*Pi*R)*Sin(3*P) + 
             12.*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*Z)))/
      (mui*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
  (beta*Pi*Cos(4*Pi*t)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*Z))/mui + 
  FELTORPERP*(0. - (0.03333333333333333*taui*Z*Sin(2*P)*Sin(2*Pi*(-10 + R))*
        Sin(2*Pi*t)*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
     (Pi*Cos(3*Pi*Z)*Sin(2*P)*Sin(3*P)*Sin(2*Pi*(-10 + R))*Sin(3*Pi*R)*
        Sin(2*Pi*t)*Sin(3*Pi*t)*Sin(2*Pi*Z))/
      (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
     (eta*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        ((Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. - 
          (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (3.*Sqrt(-mue))))/mui - 
     (nuperp*((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           3. - (8*Power(Pi,2)*R*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/3.))/R + 
     (2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        ((-0.1*R*taui*Z)/
           Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
          (3*Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t))/
           (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
          (beta*Pi*R*Cos(4*Pi*Z)*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
             Sin(4*Pi*R)*Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z))/
           (30.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))/3. + 
     ((-0.20943951023931953*R*taui*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*R)*
           Sin(2*Pi*t)*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
         Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
        (0.10471975511965977*R*taui*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
           Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
         Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (0.03333333333333333*R*taui*Z*Sin(2*P)*Sin(2*Pi*R)*Sin(2*Pi*t)*
           (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5))/
      (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     (beta*taui*Sin(P)*Sin(4*P)*Sin(Pi*t)*Sin(4*Pi*t)*
        (0.4934802200544679*R*Cos(Pi*R)*Cos(4*Pi*Z)*Sin(4*Pi*R)*Sin(Pi*Z) + 
          Cos(Pi*Z)*Sin(Pi*R)*(-0.4934802200544679*R*Cos(4*Pi*R) - 
             0.039269908169872414*Sin(4*Pi*R))*Sin(4*Pi*Z)))/
      (mui*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     (3*beta*Pi*Sin(3*P)*Sin(4*P)*Sin(3*Pi*t)*Sin(4*Pi*t)*
        (4*Pi*R*Cos(3*Pi*R)*Cos(4*Pi*Z)*Sin(4*Pi*R)*Sin(3*Pi*Z) - 
          Cos(3*Pi*Z)*Sin(3*Pi*R)*(4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*
           Sin(4*Pi*Z)))/
      (200.*mui*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
     (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
        ((taui*(-50. + 1.*R - 0.1*Power(Z,2)))/
           Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
          (-9*taui - mui*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
           (90.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
          (3*Pi*R*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (50.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
          (beta*Sin(2*P)*Sin(4*P)*Sin(2*Pi*(-10 + R))*
             (4*Pi*R*Cos(4*Pi*R) + Sin(4*Pi*R))*Sin(2*Pi*t)*Sin(4*Pi*t)*
             Sin(2*Pi*Z)*Sin(4*Pi*Z))/
           (120.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))/3.)
; }};
struct SPhie{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 0. - ((0.029608813203268074*Power(R,3)*Cos(Pi*Z)*Cos(3*Pi*Z)*Sin(P)*
        Sin(3*P)*Sin(Pi*(-10 + R))*Sin(3*Pi*(-10 + R))*Sin(Pi*t)*
        Sin(3*Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
     (0.03769911184307752*Power(R,3)*Z*Cos(3*Pi*Z)*Sin(3*P)*
        Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)))/
      Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
     (0.029608813203268074*Power(R,3)*Cos(Pi*(-10 + R))*
        Cos(3*Pi*(-10 + R))*Sin(P)*Sin(3*P)*Sin(Pi*t)*Sin(3*Pi*t)*
        Sin(Pi*Z)*Sin(3*Pi*Z))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
     (0.03769911184307752*(-10 + R)*Power(R,3)*Cos(3*Pi*(-10 + R))*
        Sin(3*P)*Sin(3*Pi*t)*(1 + 
          0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*Sin(3*Pi*Z))/
      Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
     (9*Pi*Power(R,2)*Cos(3*Pi*(-10 + R))*Sin(3*P)*Sin(3*Pi*t)*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*Sin(3*Pi*Z))/
      (500.*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
     (9*Power(Pi,2)*Power(R,3)*Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*Sin(3*Pi*Z))/
      (250.*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/R
; }};
struct SPhii{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (mui*Power(R,2)*Power(Sin(3*P),2)*Power(Sin(3*Pi*t),2)*
    (0.017765287921960846*Power(Cos(3*Pi*Z),2)*Power(Sin(3*Pi*R),2) + 
      0.017765287921960846*Power(Cos(3*Pi*R),2)*Power(Sin(3*Pi*Z),2)))/
  (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))
; }};
struct SGammaPhie{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (-0.5*mui*taui*((3*Pi*Cos(3*Pi*(-10 + R))*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
       5. - (18*Power(Pi,2)*R*Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
         Sin(3*Pi*Z))/5.))/R
; }};
struct SGammaNi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 0. - (0.5*mui*taui*(0. + 1.5707963267948966*Cos(Pi*(-10 + R))*Sin(P)*
        Sin(Pi*t)*Sin(Pi*Z) - 9.869604401089358*R*Sin(P)*Sin(Pi*(-10 + R))*
        Sin(Pi*t)*Sin(Pi*Z)))/R
; }};
struct SA{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return -(beta*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
      (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/3. \
+ (beta*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
     (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
   (3.*Sqrt(-mue)) - (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*
      Sin(4*Pi*Z) - 8*beta*Power(Pi,2)*R*Sin(4*P)*Sin(4*Pi*(-10 + R))*
      Sin(4*Pi*t)*Sin(4*Pi*Z))/R
; }};
}}//namespace feltor namespace manufactured
