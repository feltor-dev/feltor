//Mathematica generated file. Do not manually edit!
//Manufactures a solution for the feltor equations for circular flux-surfaces with R0=10, I0=20, a=1 and all Dirichlet boundary conditions
//Defines the functors: 
//    Ne, Ni, Ue, Ui, Phie, Phii, GammaPhie, GammaNi, A
//as well as
//    SNe, SNi, SUe, SUi, SPhie, SPhii, SGammaPhie, SGammaNi, SA
//Requires a definition for DG_DEVICE (define empty for host code and __host__ __device__ to generate CUDA code!
#include <cmath>
DG_DEVICE double Power(double x, double y){ return pow(x,y);}
DG_DEVICE double Sqrt(double x){ return sqrt(x);}
DG_DEVICE double Sin(double x){ return sin(x);}
DG_DEVICE double Cos(double x){ return cos(x);}
static const double Pi = M_PI;
struct Ne{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct Ni{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct Ue{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3.
; }};
struct Ui{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3.
; }};
struct Phie{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.
; }};
struct Phii{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.
; }};
struct GammaPhie{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.
; }};
struct GammaNi{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct A{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/4.
; }};
struct SNe{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1.5707963267948966*Cos(Pi*t)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*Z) - 
  (nuparallel*((-1.5707963267948966*(-10 + R)*Z*Cos(Pi*Z)*Sin(P)*
          Sin(Pi*(-10 + R))*Sin(Pi*t))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (1.5707963267948966*Power(Z,2)*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*
          Sin(Pi*Z))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
        (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
- (31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (1.5707963267948966*(-10 + R)*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (4.934802200544679*Power(-10 + R,2)*Sin(P)*Sin(Pi*(-10 + R))*
             Sin(Pi*t)*Sin(Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (200.*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))) + 
       R*((-4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
+ (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (1.5707963267948966*Z*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (4.934802200544679*Power(Z,2)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/R - 
  (nuperp*((1.5707963267948966*(-10 + R)*Z*Cos(Pi*Z)*Sin(P)*
          Sin(Pi*(-10 + R))*Sin(Pi*t))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       1.5707963267948966*(1 - 
          (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
        Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) - 
       (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
        (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
+ (31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.5707963267948966*(-10 + R)*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          4.934802200544679*(1 - 
             (1.*Power(-10 + R,2))/
              (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(P)*
           Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)) + 
       R*((31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          0.5*(Power(R,-2) - 400/
              (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
           Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)) + 
       R*((4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
- (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.5707963267948966*Z*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
          (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          4.934802200544679*(1 - 
             (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
           Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))))/R + 
  (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      ((-4.*taue*Z*(-1000. + 20.*R - 2.*Power(Z,2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (4.*taue*Z)/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (3*Pi*Cos(3*Pi*Z)*(3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
             0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (27.925268031909273*mue*(-50. + 1.*R - 0.1*Power(Z,2))*
           Cos(2*Pi*Z)*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
           Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (8.88888888888889*Z*(-50. + 1.*R - 0.1*Power(Z,2))*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (0.4444444444444445*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (0.06*Cos(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
        (2.*Z*(3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
             0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*(-10 + R)*Z)/
              Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
             (78.95683520871486*beta*Cos(4*Pi*Z)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                       0.015915494309189534*Power(R,2) - 
                       0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
             (6.283185307179586*beta*
                (2.*R*Z*Cos(4*Pi*R)*Sin(4*P) + 
                  (-0.03183098861837907*Power(Z,2)*Cos(4*P) + 
                     (-7.957747154594767 + 0.3183098861837907*R - 
                       0.015915494309189534*Power(R,2) - 
                       0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     0.15915494309189535*Z*Sin(4*P))*Sin(4*Pi*R))*
                Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
             (25.132741228718345*beta*Z*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                        0.015915494309189534*Power(R,2) - 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)))/3. + 
        (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
           ((-1.*(-10 + R))/
              Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
             (6.283185307179586*beta*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                        0.015915494309189534*Power(R,2) - 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     1.5707963267948966*R*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
      ((taue*(-1000. + 20.*R - 2.*Power(Z,2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (2.2222222222222223*(-50. + 1.*R - 0.1*Power(Z,2))*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        ((3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
             0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((-1.*(-10 + R))/
              Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
             (6.283185307179586*beta*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                        0.015915494309189534*Power(R,2) - 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      ((4.*R*(-20. + 2.*R)*taue*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (2.*taue*Z)/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (2.792526803190927*mue*R*Z*Cos(2*Pi*R)*Power(Sin(2*P),2)*
           Sin(2*Pi*R)*Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (0.4444444444444444*R*(-20. + 2.*R)*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (0.2222222222222222*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*Cos(3*Pi*Z)*Sin(3*P) - 
             0.06*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (3*Pi*Cos(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
        ((-20. + 2.*R)*Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((-1.*(-10 + R)*Z)/
              Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(-20. + 2.*R)*Cos(4*Pi*Z)*Sin(4*P) + 
                  (499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((11.140846016432674 - 0.954929658551372*R + 
                        0.0477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
             (78.95683520871486*beta*Cos(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
             (12.566370614359172*beta*(-20. + 2.*R)*Sin(4*Pi*R)*
                Sin(4*Pi*t)*(R*
                   (499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)))/3. + 
        (2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     1.5707963267948966*R*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z)*
      (0. - (2.*R*taue*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (0.2222222222222222*R*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      (0. - (2.*R*taue*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (0.2222222222222222*R*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      ((-0.044444444444444446*mue*
           (900. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(2*P)*
           Sin(2*P)*Power(Sin(2*Pi*R),2)*Power(Sin(2*Pi*t),2)*
           Power(Sin(2*Pi*Z),2))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (3*Cos(3*P)*Sin(3*Pi*t)*
           (0.1884955592153876*Z*Cos(3*Pi*Z)*Sin(3*Pi*R) + 
             (-1.884955592153876 + 0.1884955592153876*R)*Cos(3*Pi*R)*
              Sin(3*Pi*Z)))/(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) \
- (0.41887902047863906*beta*Cos(4*P)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
           Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z)*
           (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(4*Pi*Z)*
              Sin(4*Pi*R) + ((-5000. - 30.*Power(R,2) + 
                   1.*Power(R,3) - 10.*Power(Z,2) + 
                   R*(700. + 1.*Power(Z,2)))*Cos(4*Pi*R) + 
                (71.6197243913529 - 1.5915494309189535*R + 
                   0.07957747154594767*Power(R,2) + 
                   0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
              Sin(4*Pi*Z)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           (20/(R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
             (0.3141592653589793*beta*Sin(4*P)*Sin(4*Pi*t)*
                (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
                   Cos(4*Pi*Z)*Sin(4*Pi*R) + 
                  ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                        10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*
                      Cos(4*Pi*R) + 
                     (71.6197243913529 - 1.5915494309189535*R + 
                        0.07957747154594767*Power(R,2) + 
                        0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
                   Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     0.5*R*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)*
      ((taue*(250000.00000000006 - 80.*Power(R,3) + 1.*Power(R,4) + 
             1000.0000000000001*Power(Z,2) + 1.*Power(Z,4) + 
             R*(-40000. - 80.*Power(Z,2)) + 
             Power(R,2)*(2200. + 2.*Power(Z,2))))/
         (R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)) - 
        (0.011111111111111112*(900. - 20.*R + 1.*Power(R,2) + 
             1.*Power(Z,2))*(9.*taue + 
             mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*P)*Sin(3*Pi*t)*(0.1884955592153876*Z*Cos(3*Pi*Z)*
              Sin(3*Pi*R) + (-1.884955592153876 + 0.1884955592153876*R)*
              Cos(3*Pi*R)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           (20/(R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
             (0.3141592653589793*beta*Sin(4*P)*Sin(4*Pi*t)*
                (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
                   Cos(4*Pi*Z)*Sin(4*Pi*R) + 
                  ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                        10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*
                      Cos(4*Pi*R) + 
                     (71.6197243913529 - 1.5915494309189535*R + 
                        0.07957747154594767*Power(R,2) + 
                        0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
                   Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/R
; }};
struct SNi{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1.5707963267948966*Cos(Pi*t)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*Z) - 
  (nuparallel*((-1.5707963267948966*(-10 + R)*Z*Cos(Pi*Z)*Sin(P)*
          Sin(Pi*(-10 + R))*Sin(Pi*t))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (1.5707963267948966*Power(Z,2)*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*
          Sin(Pi*Z))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
        (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
- (31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (1.5707963267948966*(-10 + R)*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (4.934802200544679*Power(-10 + R,2)*Sin(P)*Sin(Pi*(-10 + R))*
             Sin(Pi*t)*Sin(Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (200.*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))) + 
       R*((-4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
+ (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (1.5707963267948966*Z*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (4.934802200544679*Power(Z,2)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/R - 
  (nuperp*((1.5707963267948966*(-10 + R)*Z*Cos(Pi*Z)*Sin(P)*
          Sin(Pi*(-10 + R))*Sin(Pi*t))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       1.5707963267948966*(1 - 
          (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
        Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) - 
       (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
        (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
+ (31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.5707963267948966*(-10 + R)*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          4.934802200544679*(1 - 
             (1.*Power(-10 + R,2))/
              (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(P)*
           Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)) + 
       R*((31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          0.5*(Power(R,-2) - 400/
              (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
           Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)) + 
       R*((4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
- (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.5707963267948966*Z*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
          (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          4.934802200544679*(1 - 
             (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
           Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))))/R + 
  (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      ((-4.*taue*Z*(-1000. + 20.*R - 2.*Power(Z,2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (4.*taue*Z)/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (3*Pi*Cos(3*Pi*Z)*(3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
             0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (27.925268031909273*mue*(-50. + 1.*R - 0.1*Power(Z,2))*
           Cos(2*Pi*Z)*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
           Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (8.88888888888889*Z*(-50. + 1.*R - 0.1*Power(Z,2))*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (0.4444444444444445*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (0.06*Cos(3*P)*Sin(3*Pi*R)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
        (2.*Z*(3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
             0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*(-10 + R)*Z)/
              Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
             (78.95683520871486*beta*Cos(4*Pi*Z)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                       0.015915494309189534*Power(R,2) - 
                       0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
             (6.283185307179586*beta*
                (2.*R*Z*Cos(4*Pi*R)*Sin(4*P) + 
                  (-0.03183098861837907*Power(Z,2)*Cos(4*P) + 
                     (-7.957747154594767 + 0.3183098861837907*R - 
                       0.015915494309189534*Power(R,2) - 
                       0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     0.15915494309189535*Z*Sin(4*P))*Sin(4*Pi*R))*
                Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
             (25.132741228718345*beta*Z*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                        0.015915494309189534*Power(R,2) - 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)))/3. + 
        (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
           ((-1.*(-10 + R))/
              Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
             (6.283185307179586*beta*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                        0.015915494309189534*Power(R,2) - 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     1.5707963267948966*R*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
      ((taue*(-1000. + 20.*R - 2.*Power(Z,2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (2.2222222222222223*(-50. + 1.*R - 0.1*Power(Z,2))*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        ((3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
             0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((-1.*(-10 + R))/
              Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
             (6.283185307179586*beta*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                  (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                        0.015915494309189534*Power(R,2) - 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                     (39.78873577297384 - 0.7957747154594768*R + 
                        0.07957747154594767*Power(Z,2))*Sin(4*P))*
                   Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      ((4.*R*(-20. + 2.*R)*taue*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (2.*taue*Z)/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (2.792526803190927*mue*R*Z*Cos(2*Pi*R)*Power(Sin(2*P),2)*
           Sin(2*Pi*R)*Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (0.4444444444444444*R*(-20. + 2.*R)*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
        (0.2222222222222222*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*Cos(3*Pi*Z)*Sin(3*P) - 
             0.06*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (3*Pi*Cos(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
        ((-20. + 2.*R)*Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((-1.*(-10 + R)*Z)/
              Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(-20. + 2.*R)*Cos(4*Pi*Z)*Sin(4*P) + 
                  (499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((11.140846016432674 - 0.954929658551372*R + 
                        0.0477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
             (78.95683520871486*beta*Cos(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
             (12.566370614359172*beta*(-20. + 2.*R)*Sin(4*Pi*R)*
                Sin(4*Pi*t)*(R*
                   (499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)))/3. + 
        (2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     1.5707963267948966*R*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z)*
      (0. - (2.*R*taue*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (0.2222222222222222*R*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      (0. - (2.*R*taue*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
        (0.2222222222222222*R*Z*
           (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*Pi*R)*Sin(3*Pi*t)*
           (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
             (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
             (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
                (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                  ((-79.57747154594767 + 11.140846016432674*R - 
                        0.477464829275686*Power(R,2) + 
                        0.015915494309189534*Power(R,3) - 
                        0.15915494309189535*Power(Z,2) + 
                        0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                     0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
      ((-0.044444444444444446*mue*
           (900. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(2*P)*
           Sin(2*P)*Power(Sin(2*Pi*R),2)*Power(Sin(2*Pi*t),2)*
           Power(Sin(2*Pi*Z),2))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (3*Cos(3*P)*Sin(3*Pi*t)*
           (0.1884955592153876*Z*Cos(3*Pi*Z)*Sin(3*Pi*R) + 
             (-1.884955592153876 + 0.1884955592153876*R)*Cos(3*Pi*R)*
              Sin(3*Pi*Z)))/(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) \
- (0.41887902047863906*beta*Cos(4*P)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
           Sin(2*Pi*t)*Sin(4*Pi*t)*Sin(2*Pi*Z)*
           (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(4*Pi*Z)*
              Sin(4*Pi*R) + ((-5000. - 30.*Power(R,2) + 
                   1.*Power(R,3) - 10.*Power(Z,2) + 
                   R*(700. + 1.*Power(Z,2)))*Cos(4*Pi*R) + 
                (71.6197243913529 - 1.5915494309189535*R + 
                   0.07957747154594767*Power(R,2) + 
                   0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
              Sin(4*Pi*Z)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           (20/(R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
             (0.3141592653589793*beta*Sin(4*P)*Sin(4*Pi*t)*
                (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
                   Cos(4*Pi*Z)*Sin(4*Pi*R) + 
                  ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                        10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*
                      Cos(4*Pi*R) + 
                     (71.6197243913529 - 1.5915494309189535*R + 
                        0.07957747154594767*Power(R,2) + 
                        0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
                   Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.) + 
     0.5*R*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)*
      ((taue*(250000.00000000006 - 80.*Power(R,3) + 1.*Power(R,4) + 
             1000.0000000000001*Power(Z,2) + 1.*Power(Z,4) + 
             R*(-40000. - 80.*Power(Z,2)) + 
             Power(R,2)*(2200. + 2.*Power(Z,2))))/
         (R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)) - 
        (0.011111111111111112*(900. - 20.*R + 1.*Power(R,2) + 
             1.*Power(Z,2))*(9.*taue + 
             mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
        (Sin(3*P)*Sin(3*Pi*t)*(0.1884955592153876*Z*Cos(3*Pi*Z)*
              Sin(3*Pi*R) + (-1.884955592153876 + 0.1884955592153876*R)*
              Cos(3*Pi*R)*Sin(3*Pi*Z)))/
         (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
           (20/(R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
             (0.3141592653589793*beta*Sin(4*P)*Sin(4*Pi*t)*
                (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
                   Cos(4*Pi*Z)*Sin(4*Pi*R) + 
                  ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                        10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*
                      Cos(4*Pi*R) + 
                     (71.6197243913529 - 1.5915494309189535*R + 
                        0.07957747154594767*Power(R,2) + 
                        0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
                   Sin(4*Pi*Z)))/
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/R
; }};
struct SUe{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 0. + (2*Pi*Cos(2*Pi*t)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*Z))/3. + 
  ((41.8879020478639*R*taue*(-50. + 1.*R - 0.1*Power(Z,2))*Cos(2*Pi*Z)*
        Sin(2*P)*Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
     (20.94395102393195*R*taue*(-50. + 1.*R - 0.1*Power(Z,2))*Cos(Pi*Z)*
        Sin(P)*Sin(2*P)*Sin(Pi*R)*Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*
        Sin(2*Pi*Z))/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) \
- (2.0943951023931953*Power(R,2)*taue*Z*Cos(Pi*R)*Sin(P)*Sin(2*P)*
        Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
     (0.03333333333333333*R*taue*
        (900. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(P)*Sin(2*P)*
        Sin(Pi*R)*Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
     (4.1887902047863905*Power(R,2)*taue*Z*Cos(2*Pi*R)*Sin(2*P)*
        Sin(2*Pi*t)*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
        Sin(2*Pi*Z))/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) \
- (0.06666666666666667*R*taue*
        (900. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(2*P)*
        Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
     (1.3333333333333333*Power(R,2)*(-20. + 2.*R)*taue*Z*Sin(2*P)*
        Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
     (26.666666666666664*R*taue*Z*(-50. + 1.*R - 0.1*Power(Z,2))*Sin(2*P)*
        Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
     (2.6666666666666665*R*taue*Z*Sin(2*P)*Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2))/
   (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) - 
  (nuparallel*((-2.0943951023931953*(-10 + R)*Z*Cos(2*Pi*Z)*Sin(2*P)*
          Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (2.0943951023931953*Power(Z,2)*Cos(2*Pi*(-10 + R))*Sin(2*P)*
          Sin(2*Pi*t)*Sin(2*Pi*Z))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (13.333333333333332*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          Sin(2*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-13.15947253478581*(-10 + R)*Z*Cos(2*Pi*(-10 + R))*
             Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (83.7758040957278*(-10 + R)*Cos(2*P)*Cos(2*Pi*Z)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (4.1887902047863905*Power(-10 + R,2)*Z*Cos(2*Pi*Z)*Sin(2*P)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (4.1887902047863905*(-10 + R)*Power(Z,2)*Cos(2*Pi*(-10 + R))*
             Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (2.0943951023931953*(-10 + R)*Cos(2*Pi*(-10 + R))*Sin(2*P)*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (26.666666666666664*(-10 + R)*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (13.15947253478581*Power(-10 + R,2)*Sin(2*P)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-83.7758040957278*(-10 + R)*Cos(2*P)*Cos(2*Pi*Z)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (83.7758040957278*Z*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) \
- (1600*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (3.*Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))) + 
       R*((-13.15947253478581*(-10 + R)*Z*Cos(2*Pi*(-10 + R))*Cos(2*Pi*Z)*
             Sin(2*P)*Sin(2*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (4.1887902047863905*Power(-10 + R,2)*Z*Cos(2*Pi*Z)*Sin(2*P)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (2.0943951023931953*Z*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (83.7758040957278*Z*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (4.1887902047863905*(-10 + R)*Power(Z,2)*Cos(2*Pi*(-10 + R))*
             Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (26.666666666666664*(-10 + R)*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (13.333333333333332*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (13.15947253478581*Power(Z,2)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/R + 
  (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(3*Pi*t)*Sin(2*Pi*Z)*
     ((-1884.9555921538758 + 37.69911184307752*R - 
          3.7699111843077526*Power(Z,2))*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R) + 
       (-3.7699111843077517*R*Z*Cos(3*Pi*R)*Sin(3*P) + 
          (-54. + 1.2000000000000002*R - 0.06000000000000001*Power(R,2) - 
             0.06000000000000001*Power(Z,2))*Cos(3*P)*Sin(3*Pi*R))*
        Sin(3*Pi*Z)))/
   (3.*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)) + 
  (beta*Pi*Cos(4*Pi*t)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*Z))/mue + 
  (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
     ((taue*(-1000. + 20.*R - 2.*Power(Z,2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       (2.2222222222222223*(-50. + 1.*R - 0.1*Power(Z,2))*
          (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
             Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       ((3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
            0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
       (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
          ((-1.*(-10 + R))/
             Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
            (6.283185307179586*beta*
               (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                    1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                 (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                       0.015915494309189534*Power(R,2) - 
                       0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                    (39.78873577297384 - 0.7957747154594768*R + 
                       0.07957747154594767*Power(Z,2))*Sin(4*P))*
                  Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
             Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/3. \
+ (2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z)*
     (0. - (2.*R*taue*Z)/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
       (0.2222222222222222*R*Z*
          (9.*taue + mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
             Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       (Sin(3*Pi*R)*Sin(3*Pi*t)*
          (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
            (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
       (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
          ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
            (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
               (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                    1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                 ((-79.57747154594767 + 11.140846016432674*R - 
                       0.477464829275686*Power(R,2) + 
                       0.015915494309189534*Power(R,3) - 
                       0.15915494309189535*Power(Z,2) + 
                       0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                    0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
             Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/3. \
+ (1.5707963267948966*taue*Sin(Pi*t)*
     (1.*R*Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*
        ((10. - 1.*R)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
            1.5) - 6.283185307179586*beta*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
             (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                   0.015915494309189534*Power(R,2) - 
                   0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                (39.78873577297384 - 0.7957747154594768*R + 
                   0.07957747154594767*Power(Z,2))*Sin(4*P))*Sin(4*Pi*R))*
           Sin(4*Pi*t)*Sin(4*Pi*Z)) + 
       1.*R*Cos(Pi*R)*Sin(P)*Sin(Pi*Z)*
        (1.*Z*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
          6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
             ((-79.57747154594767 + 11.140846016432674*R - 
                   0.477464829275686*Power(R,2) + 
                   0.015915494309189534*Power(R,3) - 
                   0.15915494309189535*Power(Z,2) + 
                   0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z))) + 
       0.1*Cos(P)*Sin(Pi*R)*Sin(Pi*Z)*
        (63.66197723675813*Power(500. - 20.*R + 1.*Power(R,2) + 
             1.*Power(Z,2),1.5) - 
          1.*beta*R*Sin(4*P)*Sin(4*Pi*t)*
           (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(4*Pi*Z)*
              Sin(4*Pi*R) + ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                   10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*Cos(4*Pi*R) \
+ (71.6197243913529 - 1.5915494309189535*R + 
                   0.07957747154594767*Power(R,2) + 
                   0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*Sin(4*Pi*Z))\
)))/(mue*R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)*
     (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
  (3*Sin(3*Pi*t)*(Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*
        ((10. - 1.*R)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
            1.5) - 6.283185307179586*beta*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
             (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                   0.015915494309189534*Power(R,2) - 
                   0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                (39.78873577297384 - 0.7957747154594768*R + 
                   0.07957747154594767*Power(Z,2))*Sin(4*P))*Sin(4*Pi*R))*
           Sin(4*Pi*t)*Sin(4*Pi*Z)) + 
       3.141592653589793*R*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*Z)*
        (1.*Z*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
          6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
             ((-79.57747154594767 + 11.140846016432674*R - 
                   0.477464829275686*Power(R,2) + 
                   0.015915494309189534*Power(R,3) - 
                   0.15915494309189535*Power(Z,2) + 
                   0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z))) + 
       0.3141592653589793*Cos(3*P)*Sin(3*Pi*R)*Sin(3*Pi*Z)*
        (63.66197723675813*Power(500. - 20.*R + 1.*Power(R,2) + 
             1.*Power(Z,2),1.5) - 
          1.*beta*R*Sin(4*P)*Sin(4*Pi*t)*
           (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(4*Pi*Z)*
              Sin(4*Pi*R) + ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                   10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*Cos(4*Pi*R) \
+ (71.6197243913529 - 1.5915494309189535*R + 
                   0.07957747154594767*Power(R,2) + 
                   0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*Sin(4*Pi*Z))\
)))/(5.*mue*R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)) + 
  (2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
     ((taue*(250000.00000000006 - 80.*Power(R,3) + 1.*Power(R,4) + 
            1000.0000000000001*Power(Z,2) + 1.*Power(Z,4) + 
            R*(-40000. - 80.*Power(Z,2)) + 
            Power(R,2)*(2200. + 2.*Power(Z,2))))/
        (R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)) - 
       (0.011111111111111112*(900. - 20.*R + 1.*Power(R,2) + 
            1.*Power(Z,2))*(9.*taue + 
            mue*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
             Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       (Sin(3*P)*Sin(3*Pi*t)*(0.1884955592153876*Z*Cos(3*Pi*Z)*
             Sin(3*Pi*R) + (-1.884955592153876 + 0.1884955592153876*R)*
             Cos(3*Pi*R)*Sin(3*Pi*Z)))/
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
       (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
          (20/(R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
            (0.3141592653589793*beta*Sin(4*P)*Sin(4*Pi*t)*
               (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
                  Cos(4*Pi*Z)*Sin(4*Pi*R) + 
                 ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                       10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*
                     Cos(4*Pi*R) + 
                    (71.6197243913529 - 1.5915494309189535*R + 
                       0.07957747154594767*Power(R,2) + 
                       0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
                  Sin(4*Pi*Z)))/
             Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/3. \
- (nuperp*((1.*(-10 + R)*Z*((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
               Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/3. + 
            (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t))/
             mue))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (1 - (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
        ((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. + 
          (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*Sin(4*Pi*Z))/mue\
) - (20.*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. + 
            (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/mue\
))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((20.*(-10 + R)*((4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*Z)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mue))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (20.*Z*((4*Pi*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mue))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (Power(R,-2) - 400/
              (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
           ((-4*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. - 
             (4*beta*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              mue)) + R*((1.*(-10 + R)*Z*
             ((4*Power(Pi,2)*Cos(2*Pi*(-10 + R))*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*t))/3. + 
               (4*beta*Power(Pi,2)*Cos(4*Pi*(-10 + R))*Cos(4*Pi*Z)*
                  Sin(4*P)*Sin(4*Pi*t))/mue))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20.*(-10 + R)*((4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*Z)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mue))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (2.*Power(-10 + R,2)*Z*
             ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t))/3. + 
               (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mue))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (2.*(-10 + R)*Power(Z,2)*
             ((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mue))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.*(-10 + R)*((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mue))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (40.*(-10 + R)*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mue))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
          (1 - (1.*Power(-10 + R,2))/
              (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
           ((-4*Power(Pi,2)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                Sin(2*Pi*Z))/3. - 
             (4*beta*Power(Pi,2)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                Sin(4*Pi*Z))/mue)) + 
       R*((1.*(-10 + R)*Z*((4*Power(Pi,2)*Cos(2*Pi*(-10 + R))*
                  Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/3. + 
               (4*beta*Power(Pi,2)*Cos(4*Pi*(-10 + R))*Cos(4*Pi*Z)*
                  Sin(4*P)*Sin(4*Pi*t))/mue))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (2.*Power(-10 + R,2)*Z*
             ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                3. + (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mue))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.*Z*((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t))/3. + 
               (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mue))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (20.*Z*((4*Pi*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mue))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (2.*(-10 + R)*Power(Z,2)*
             ((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                3. + (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mue))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (40.*(-10 + R)*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
                mue))/(R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) \
+ (20.*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. + 
               (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
                mue))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (1 - (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
           ((-4*Power(Pi,2)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                Sin(2*Pi*Z))/3. - 
             (4*beta*Power(Pi,2)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                Sin(4*Pi*Z))/mue))))/R
; }};
struct SUi{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 0. + (2*Pi*Cos(2*Pi*t)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*Z))/3. + 
  ((41.8879020478639*R*taui*(-50. + 1.*R - 0.1*Power(Z,2))*Cos(2*Pi*Z)*
        Sin(2*P)*Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
     (20.94395102393195*R*taui*(-50. + 1.*R - 0.1*Power(Z,2))*Cos(Pi*Z)*
        Sin(P)*Sin(2*P)*Sin(Pi*R)*Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*
        Sin(2*Pi*Z))/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) \
- (2.0943951023931953*Power(R,2)*taui*Z*Cos(Pi*R)*Sin(P)*Sin(2*P)*
        Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
     (0.03333333333333333*R*taui*
        (900. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(P)*Sin(2*P)*
        Sin(Pi*R)*Sin(2*Pi*R)*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
     (4.1887902047863905*Power(R,2)*taui*Z*Cos(2*Pi*R)*Sin(2*P)*
        Sin(2*Pi*t)*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
        Sin(2*Pi*Z))/Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) \
- (0.06666666666666667*R*taui*
        (900. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(2*P)*
        Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
     (1.3333333333333333*Power(R,2)*(-20. + 2.*R)*taui*Z*Sin(2*P)*
        Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
     (26.666666666666664*R*taui*Z*(-50. + 1.*R - 0.1*Power(Z,2))*Sin(2*P)*
        Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3) - 
     (2.6666666666666665*R*taui*Z*Sin(2*P)*Sin(2*Pi*R)*Sin(2*Pi*t)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
      Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2))/
   (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) - 
  (nuparallel*((-2.0943951023931953*(-10 + R)*Z*Cos(2*Pi*Z)*Sin(2*P)*
          Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (2.0943951023931953*Power(Z,2)*Cos(2*Pi*(-10 + R))*Sin(2*P)*
          Sin(2*Pi*t)*Sin(2*Pi*Z))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (13.333333333333332*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
          Sin(2*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-13.15947253478581*(-10 + R)*Z*Cos(2*Pi*(-10 + R))*
             Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (83.7758040957278*(-10 + R)*Cos(2*P)*Cos(2*Pi*Z)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (4.1887902047863905*Power(-10 + R,2)*Z*Cos(2*Pi*Z)*Sin(2*P)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (4.1887902047863905*(-10 + R)*Power(Z,2)*Cos(2*Pi*(-10 + R))*
             Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (2.0943951023931953*(-10 + R)*Cos(2*Pi*(-10 + R))*Sin(2*P)*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (26.666666666666664*(-10 + R)*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (13.15947253478581*Power(-10 + R,2)*Sin(2*P)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((-83.7758040957278*(-10 + R)*Cos(2*P)*Cos(2*Pi*Z)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (83.7758040957278*Z*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) \
- (1600*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (3.*Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))) + 
       R*((-13.15947253478581*(-10 + R)*Z*Cos(2*Pi*(-10 + R))*Cos(2*Pi*Z)*
             Sin(2*P)*Sin(2*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (4.1887902047863905*Power(-10 + R,2)*Z*Cos(2*Pi*Z)*Sin(2*P)*
             Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (2.0943951023931953*Z*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (83.7758040957278*Z*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (4.1887902047863905*(-10 + R)*Power(Z,2)*Cos(2*Pi*(-10 + R))*
             Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (26.666666666666664*(-10 + R)*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (13.333333333333332*Z*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (13.15947253478581*Power(Z,2)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/R + 
  (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(3*Pi*t)*Sin(2*Pi*Z)*
     ((-1884.9555921538758 + 37.69911184307752*R - 
          3.7699111843077526*Power(Z,2))*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R) + 
       (-3.7699111843077517*R*Z*Cos(3*Pi*R)*Sin(3*P) + 
          (-54. + 1.2000000000000002*R - 0.06000000000000001*Power(R,2) - 
             0.06000000000000001*Power(Z,2))*Cos(3*P)*Sin(3*Pi*R))*
        Sin(3*Pi*Z)))/
   (3.*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)) + 
  (beta*Pi*Cos(4*Pi*t)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*Z))/mui + 
  (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
     ((taui*(-1000. + 20.*R - 2.*Power(Z,2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       (2.2222222222222223*(-50. + 1.*R - 0.1*Power(Z,2))*
          (9.*taui + mui*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
             Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       ((3.7699111843077517*R*Cos(3*Pi*R)*Sin(3*P) - 
            0.06*Z*Cos(3*P)*Sin(3*Pi*R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
       (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
          ((-1.*(-10 + R))/
             Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
            (6.283185307179586*beta*
               (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                    1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
                 (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                       0.015915494309189534*Power(R,2) - 
                       0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                    (39.78873577297384 - 0.7957747154594768*R + 
                       0.07957747154594767*Power(Z,2))*Sin(4*P))*
                  Sin(4*Pi*R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
             Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/3. \
+ (2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z)*
     (0. - (2.*R*taui*Z)/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) - 
       (0.2222222222222222*R*Z*
          (9.*taui + mui*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
             Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       (Sin(3*Pi*R)*Sin(3*Pi*t)*
          (-3.7699111843077517*R*Cos(3*Pi*Z)*Sin(3*P) + 
            (0.6 - 0.06*R)*Cos(3*P)*Sin(3*Pi*Z)))/
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
       (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
          ((1.*Z)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
            (6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
               (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                    1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
                 ((-79.57747154594767 + 11.140846016432674*R - 
                       0.477464829275686*Power(R,2) + 
                       0.015915494309189534*Power(R,3) - 
                       0.15915494309189535*Power(Z,2) + 
                       0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                    0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z)))/
             Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/3. \
+ (1.5707963267948966*taui*Sin(Pi*t)*
     (1.*R*Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*
        ((10. - 1.*R)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
            1.5) - 6.283185307179586*beta*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
             (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                   0.015915494309189534*Power(R,2) - 
                   0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                (39.78873577297384 - 0.7957747154594768*R + 
                   0.07957747154594767*Power(Z,2))*Sin(4*P))*Sin(4*Pi*R))*
           Sin(4*Pi*t)*Sin(4*Pi*Z)) + 
       1.*R*Cos(Pi*R)*Sin(P)*Sin(Pi*Z)*
        (1.*Z*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
          6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
             ((-79.57747154594767 + 11.140846016432674*R - 
                   0.477464829275686*Power(R,2) + 
                   0.015915494309189534*Power(R,3) - 
                   0.15915494309189535*Power(Z,2) + 
                   0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z))) + 
       0.1*Cos(P)*Sin(Pi*R)*Sin(Pi*Z)*
        (63.66197723675813*Power(500. - 20.*R + 1.*Power(R,2) + 
             1.*Power(Z,2),1.5) - 
          1.*beta*R*Sin(4*P)*Sin(4*Pi*t)*
           (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(4*Pi*Z)*
              Sin(4*Pi*R) + ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                   10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*Cos(4*Pi*R) \
+ (71.6197243913529 - 1.5915494309189535*R + 
                   0.07957747154594767*Power(R,2) + 
                   0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*Sin(4*Pi*Z))\
)))/(mui*R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)*
     (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
  (3*Sin(3*Pi*t)*(Pi*R*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*R)*
        ((10. - 1.*R)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
            1.5) - 6.283185307179586*beta*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*R)*Sin(4*P) + 
             (Z*(-7.957747154594767 + 0.3183098861837907*R - 
                   0.015915494309189534*Power(R,2) - 
                   0.015915494309189534*Power(Z,2))*Cos(4*P) + 
                (39.78873577297384 - 0.7957747154594768*R + 
                   0.07957747154594767*Power(Z,2))*Sin(4*P))*Sin(4*Pi*R))*
           Sin(4*Pi*t)*Sin(4*Pi*Z)) + 
       3.141592653589793*R*Cos(3*Pi*R)*Sin(3*P)*Sin(3*Pi*Z)*
        (1.*Z*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
          6.283185307179586*beta*Sin(4*Pi*R)*Sin(4*Pi*t)*
           (R*(499.99999999999994 - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*Cos(4*Pi*Z)*Sin(4*P) + 
             ((-79.57747154594767 + 11.140846016432674*R - 
                   0.477464829275686*Power(R,2) + 
                   0.015915494309189534*Power(R,3) - 
                   0.15915494309189535*Power(Z,2) + 
                   0.015915494309189534*R*Power(Z,2))*Cos(4*P) - 
                0.07957747154594767*R*Z*Sin(4*P))*Sin(4*Pi*Z))) + 
       0.3141592653589793*Cos(3*P)*Sin(3*Pi*R)*Sin(3*Pi*Z)*
        (63.66197723675813*Power(500. - 20.*R + 1.*Power(R,2) + 
             1.*Power(Z,2),1.5) - 
          1.*beta*R*Sin(4*P)*Sin(4*Pi*t)*
           (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(4*Pi*Z)*
              Sin(4*Pi*R) + ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                   10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*Cos(4*Pi*R) \
+ (71.6197243913529 - 1.5915494309189535*R + 
                   0.07957747154594767*Power(R,2) + 
                   0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*Sin(4*Pi*Z))\
)))/(5.*mui*R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)) + 
  (2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
     ((taui*(250000.00000000006 - 80.*Power(R,3) + 1.*Power(R,4) + 
            1000.0000000000001*Power(Z,2) + 1.*Power(Z,4) + 
            R*(-40000. - 80.*Power(Z,2)) + 
            Power(R,2)*(2200. + 2.*Power(Z,2))))/
        (R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),3)) - 
       (0.011111111111111112*(900. - 20.*R + 1.*Power(R,2) + 
            1.*Power(Z,2))*(9.*taui + 
            mui*Power(Sin(2*P),2)*Power(Sin(2*Pi*R),2)*
             Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2)))/
        Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2) + 
       (Sin(3*P)*Sin(3*Pi*t)*(0.1884955592153876*Z*Cos(3*Pi*Z)*
             Sin(3*Pi*R) + (-1.884955592153876 + 0.1884955592153876*R)*
             Cos(3*Pi*R)*Sin(3*Pi*Z)))/
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
       (Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
          (20/(R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
            (0.3141592653589793*beta*Sin(4*P)*Sin(4*Pi*t)*
               (Z*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
                  Cos(4*Pi*Z)*Sin(4*Pi*R) + 
                 ((-5000. - 30.*Power(R,2) + 1.*Power(R,3) - 
                       10.*Power(Z,2) + R*(700. + 1.*Power(Z,2)))*
                     Cos(4*Pi*R) + 
                    (71.6197243913529 - 1.5915494309189535*R + 
                       0.07957747154594767*Power(R,2) + 
                       0.07957747154594767*Power(Z,2))*Sin(4*Pi*R))*
                  Sin(4*Pi*Z)))/
             Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)))/3.))/3. \
- (nuperp*((1.*(-10 + R)*Z*((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
               Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/3. + 
            (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t))/
             mui))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       (1 - (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
        ((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. + 
          (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*Sin(4*Pi*Z))/mui\
) - (20.*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. + 
            (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/mui\
))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((20.*(-10 + R)*((4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*Z)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mui))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (20.*Z*((4*Pi*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mui))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (Power(R,-2) - 400/
              (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
           ((-4*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. - 
             (4*beta*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
              mui)) + R*((1.*(-10 + R)*Z*
             ((4*Power(Pi,2)*Cos(2*Pi*(-10 + R))*Cos(2*Pi*Z)*Sin(2*P)*
                  Sin(2*Pi*t))/3. + 
               (4*beta*Power(Pi,2)*Cos(4*Pi*(-10 + R))*Cos(4*Pi*Z)*
                  Sin(4*P)*Sin(4*Pi*t))/mui))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20.*(-10 + R)*((4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                  Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*Z)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mui))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (2.*Power(-10 + R,2)*Z*
             ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t))/3. + 
               (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mui))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (2.*(-10 + R)*Power(Z,2)*
             ((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mui))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.*(-10 + R)*((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mui))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (40.*(-10 + R)*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mui))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
          (1 - (1.*Power(-10 + R,2))/
              (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
           ((-4*Power(Pi,2)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                Sin(2*Pi*Z))/3. - 
             (4*beta*Power(Pi,2)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                Sin(4*Pi*Z))/mui)) + 
       R*((1.*(-10 + R)*Z*((4*Power(Pi,2)*Cos(2*Pi*(-10 + R))*
                  Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*t))/3. + 
               (4*beta*Power(Pi,2)*Cos(4*Pi*(-10 + R))*Cos(4*Pi*Z)*
                  Sin(4*P)*Sin(4*Pi*t))/mui))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (2.*Power(-10 + R,2)*Z*
             ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t))/
                3. + (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mui))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.*Z*((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*Sin(2*Pi*(-10 + R))*
                  Sin(2*Pi*t))/3. + 
               (beta*Pi*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
                  Sin(4*Pi*t))/mui))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (20.*Z*((4*Pi*Cos(2*P)*Cos(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (4*beta*Pi*Cos(4*P)*Cos(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mui))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (2.*(-10 + R)*Power(Z,2)*
             ((2*Pi*Cos(2*Pi*(-10 + R))*Sin(2*P)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                3. + (beta*Pi*Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*
                  Sin(4*Pi*Z))/mui))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (40.*(-10 + R)*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                  Sin(2*Pi*Z))/3. + 
               (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
                mui))/(R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) \
+ (20.*Z*((2*Cos(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/3. + 
               (beta*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
                mui))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (1 - (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
           ((-4*Power(Pi,2)*Sin(2*P)*Sin(2*Pi*(-10 + R))*Sin(2*Pi*t)*
                Sin(2*Pi*Z))/3. - 
             (4*beta*Power(Pi,2)*Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
                Sin(4*Pi*Z))/mui))))/R
; }};
struct SPhie{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 0. - ((0.015707963267948967*Power(R,3)*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*
        Sin(Pi*t)*((3*Pi*(1 - 
               (1.*Power(-10 + R,2))/
                (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Cos(3*Pi*Z)*
             Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/5. + 
          (1.8849555921538759*(-10 + R)*Z*Cos(3*Pi*(-10 + R))*Sin(3*P)*
             Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (12.*(-10 + R)*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
             Sin(3*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
      (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
     (0.02*Power(R,3)*Z*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
           Sin(Pi*Z))*((3*Pi*(1 - 
               (1.*Power(-10 + R,2))/
                (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Cos(3*Pi*Z)*
             Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/5. + 
          (1.8849555921538759*(-10 + R)*Z*Cos(3*Pi*(-10 + R))*Sin(3*P)*
             Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (12.*(-10 + R)*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
             Sin(3*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
      Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
     (0.015707963267948967*Power(R,3)*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*
        Sin(Pi*Z)*((1.8849555921538759*(-10 + R)*Z*Cos(3*Pi*Z)*Sin(3*P)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (3*Pi*(1 - (1.*Power(Z,2))/
                (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
             Cos(3*Pi*(-10 + R))*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/5. - 
          (12.*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
      (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
     (0.02*(-10 + R)*Power(R,3)*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        ((1.8849555921538759*(-10 + R)*Z*Cos(3*Pi*Z)*Sin(3*P)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (3*Pi*(1 - (1.*Power(Z,2))/
                (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
             Cos(3*Pi*(-10 + R))*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/5. - 
          (12.*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
      Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
     (3*Power(R,2)*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        ((1.8849555921538759*(-10 + R)*Z*Cos(3*Pi*Z)*Sin(3*P)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (3*Pi*(1 - (1.*Power(Z,2))/
                (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
             Cos(3*Pi*(-10 + R))*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/5. - 
          (12.*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
      (100.*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
     (0.005*Power(R,3)*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)*
        ((37.69911184307752*(-10 + R)*Cos(3*Pi*Z)*Sin(3*P)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (37.69911184307752*Z*Cos(3*Pi*(-10 + R))*Sin(3*P)*Sin(3*Pi*t)*
             Sin(3*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) \
+ (3*(Power(R,-2) - 400/
                (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
             Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.))/
      (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
     (Power(R,3)*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        ((17.765287921960844*(-10 + R)*Z*Cos(3*Pi*(-10 + R))*Cos(3*Pi*Z)*
             Sin(3*P)*Sin(3*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (113.09733552923255*(-10 + R)*Cos(3*P)*Cos(3*Pi*Z)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.7699111843077517*Power(-10 + R,2)*Z*Cos(3*Pi*Z)*Sin(3*P)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (3.7699111843077517*(-10 + R)*Power(Z,2)*Cos(3*Pi*(-10 + R))*
             Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.8849555921538759*(-10 + R)*Cos(3*Pi*(-10 + R))*Sin(3*P)*
             Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (24.*(-10 + R)*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
             Sin(3*Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          (9*Power(Pi,2)*(1 - 
               (1.*Power(-10 + R,2))/
                (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(3*P)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.))/
      (100.*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
     (Power(R,3)*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        ((113.09733552923255*(-10 + R)*Cos(3*P)*Cos(3*Pi*Z)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (113.09733552923255*Z*Cos(3*P)*Cos(3*Pi*(-10 + R))*Sin(3*Pi*t)*
             Sin(3*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) \
- (9*(Power(R,-2) - 400/
                (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
             Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.))/
      (100.*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
     (Power(R,3)*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        ((17.765287921960844*(-10 + R)*Z*Cos(3*Pi*(-10 + R))*Cos(3*Pi*Z)*
             Sin(3*P)*Sin(3*Pi*t))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (3.7699111843077517*Power(-10 + R,2)*Z*Cos(3*Pi*Z)*Sin(3*P)*
             Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.8849555921538759*Z*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*(-10 + R))*
             Sin(3*Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (113.09733552923255*Z*Cos(3*P)*Cos(3*Pi*(-10 + R))*Sin(3*Pi*t)*
             Sin(3*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.7699111843077517*(-10 + R)*Power(Z,2)*Cos(3*Pi*(-10 + R))*
             Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (24.*(-10 + R)*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
             Sin(3*Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
          (12.*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (9*Power(Pi,2)*(1 - 
               (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))\
)*Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.))/
      (100.*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/R
; }};
struct SPhii{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (0.017765287921960846*mui*Power(Sin(3*Pi*t),2)*
    (Power(R,2)*(400.00000000000006 + 1.0000000000000002*Power(Z,2))*
       Power(Cos(3*Pi*Z),2)*Power(Sin(3*P),2)*Power(Sin(3*Pi*R),2) + 
      Power(R,2)*(500. - 20.000000000000004*R + 1.*Power(R,2))*
       Power(Cos(3*Pi*R),2)*Power(Sin(3*P),2)*Power(Sin(3*Pi*Z),2) + 
      10.13211836423378*Power(Cos(3*P),2)*Power(Sin(3*Pi*R),2)*
       Power(Sin(3*Pi*Z),2) - 2.026423672846756*R*Power(Cos(3*P),2)*
       Power(Sin(3*Pi*R),2)*Power(Sin(3*Pi*Z),2) + 
      0.1013211836423378*Power(R,2)*Power(Cos(3*P),2)*Power(Sin(3*Pi*R),2)*
       Power(Sin(3*Pi*Z),2) + 0.1013211836423378*Power(Z,2)*
       Power(Cos(3*P),2)*Power(Sin(3*Pi*R),2)*Power(Sin(3*Pi*Z),2) - 
      3.1830988618379075*R*Z*Sin(6*P)*Sin(6*Pi*R)*Power(Sin(3*Pi*Z),2) - 
      31.830988618379074*R*Sin(6*P)*Power(Sin(3*Pi*R),2)*Sin(6*Pi*Z) + 
      3.1830988618379075*Power(R,2)*Sin(6*P)*Power(Sin(3*Pi*R),2)*
       Sin(6*Pi*Z) - 5.*Power(R,2)*Z*Power(Sin(3*P),2)*Sin(6*Pi*R)*
       Sin(6*Pi*Z) + 0.5000000000000001*Power(R,3)*Z*Power(Sin(3*P),2)*
       Sin(6*Pi*R)*Sin(6*Pi*Z)))/
  Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)
; }};
struct SGammaPhie{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (-0.5*mui*taui*((1.8849555921538759*(-10 + R)*Z*Cos(3*Pi*Z)*Sin(3*P)*
         Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
       (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
      (3*Pi*(1 - (1.*Power(Z,2))/
            (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
         Cos(3*Pi*(-10 + R))*Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/5. - 
      (12.*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
       (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
      R*((17.765287921960844*(-10 + R)*Z*Cos(3*Pi*(-10 + R))*Cos(3*Pi*Z)*
            Sin(3*P)*Sin(3*Pi*t))/
          (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
         (113.09733552923255*(-10 + R)*Cos(3*P)*Cos(3*Pi*Z)*
            Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
          (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
         (3.7699111843077517*Power(-10 + R,2)*Z*Cos(3*Pi*Z)*Sin(3*P)*
            Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
         (3.7699111843077517*(-10 + R)*Power(Z,2)*Cos(3*Pi*(-10 + R))*
            Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
         (1.8849555921538759*(-10 + R)*Cos(3*Pi*(-10 + R))*Sin(3*P)*
            Sin(3*Pi*t)*Sin(3*Pi*Z))/
          (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
         (24.*(-10 + R)*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
            Sin(3*Pi*Z))/
          (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
         (9*Power(Pi,2)*(1 - (1.*Power(-10 + R,2))/
               (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(3*P)*
            Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.) + 
      R*((113.09733552923255*(-10 + R)*Cos(3*P)*Cos(3*Pi*Z)*
            Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
          (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
         (113.09733552923255*Z*Cos(3*P)*Cos(3*Pi*(-10 + R))*Sin(3*Pi*t)*
            Sin(3*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
         (9*(Power(R,-2) - 400/
               (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
            Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.) + 
      R*((17.765287921960844*(-10 + R)*Z*Cos(3*Pi*(-10 + R))*Cos(3*Pi*Z)*
            Sin(3*P)*Sin(3*Pi*t))/
          (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
         (3.7699111843077517*Power(-10 + R,2)*Z*Cos(3*Pi*Z)*Sin(3*P)*
            Sin(3*Pi*(-10 + R))*Sin(3*Pi*t))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
         (1.8849555921538759*Z*Cos(3*Pi*Z)*Sin(3*P)*Sin(3*Pi*(-10 + R))*
            Sin(3*Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
         (113.09733552923255*Z*Cos(3*P)*Cos(3*Pi*(-10 + R))*Sin(3*Pi*t)*
            Sin(3*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
         (3.7699111843077517*(-10 + R)*Power(Z,2)*Cos(3*Pi*(-10 + R))*
            Sin(3*P)*Sin(3*Pi*t)*Sin(3*Pi*Z))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
         (24.*(-10 + R)*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*
            Sin(3*Pi*Z))/
          (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
         (12.*Z*Cos(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
          (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
         (9*Power(Pi,2)*(1 - (1.*Power(Z,2))/
               (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(3*P)*
            Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.)))/R
; }};
struct SGammaNi{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 0. - (0.5*mui*taui*((1.5707963267948966*(-10 + R)*Z*Cos(Pi*Z)*Sin(P)*
          Sin(Pi*(-10 + R))*Sin(Pi*t))/
        (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
       1.5707963267948966*(1 - 
          (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
        Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) - 
       (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
        (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
       R*((4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) \
+ (31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*
             Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.5707963267948966*(-10 + R)*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z))/
           (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
          4.934802200544679*(1 - 
             (1.*Power(-10 + R,2))/
              (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(P)*
           Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)) + 
       R*((31.41592653589793*(-10 + R)*Cos(P)*Cos(Pi*Z)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          0.5*(Power(R,-2) - 400/
              (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
           Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)) + 
       R*((4.934802200544679*(-10 + R)*Z*Cos(Pi*(-10 + R))*Cos(Pi*Z)*
             Sin(P)*Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (3.141592653589793*Power(-10 + R,2)*Z*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*(-10 + R))*Sin(Pi*t))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (1.5707963267948966*Z*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*
             Sin(Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (31.41592653589793*Z*Cos(P)*Cos(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
          (3.141592653589793*(-10 + R)*Power(Z,2)*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z))/
           Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
          (20.*(-10 + R)*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
          (10.*Z*Cos(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))/
           (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
          4.934802200544679*(1 - 
             (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
           Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))))/R
; }};
struct SA{
    double mue,mui,taue,taui,beta,nuperp,nuparallel;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return -(((3.141592653589793*(-10 + R)*Z*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
         Sin(4*Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
      Pi*(1 - (1.*Power(Z,2))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*
       Cos(4*Pi*(-10 + R))*Sin(4*P)*Sin(4*Pi*t)*Sin(4*Pi*Z) - 
      (20.*Z*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
       (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
      R*((39.47841760435743*(-10 + R)*Z*Cos(4*Pi*(-10 + R))*Cos(4*Pi*Z)*
            Sin(4*P)*Sin(4*Pi*t))/
          (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
         (251.32741228718345*(-10 + R)*Cos(4*P)*Cos(4*Pi*Z)*
            Sin(4*Pi*(-10 + R))*Sin(4*Pi*t))/
          (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
         (6.283185307179586*Power(-10 + R,2)*Z*Cos(4*Pi*Z)*Sin(4*P)*
            Sin(4*Pi*(-10 + R))*Sin(4*Pi*t))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) - 
         (6.283185307179586*(-10 + R)*Power(Z,2)*Cos(4*Pi*(-10 + R))*
            Sin(4*P)*Sin(4*Pi*t)*Sin(4*Pi*Z))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
         (3.141592653589793*(-10 + R)*Cos(4*Pi*(-10 + R))*Sin(4*P)*
            Sin(4*Pi*t)*Sin(4*Pi*Z))/
          (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
         (40.*(-10 + R)*Z*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
            Sin(4*Pi*Z))/
          (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) - 
         4*Power(Pi,2)*(1 - (1.*Power(-10 + R,2))/
             (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(4*P)*
          Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z)) + 
      R*((251.32741228718345*(-10 + R)*Cos(4*P)*Cos(4*Pi*Z)*
            Sin(4*Pi*(-10 + R))*Sin(4*Pi*t))/
          (R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
         (251.32741228718345*Z*Cos(4*P)*Cos(4*Pi*(-10 + R))*Sin(4*Pi*t)*
            Sin(4*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
         4*(Power(R,-2) - 400/
             (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))*
          Sin(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z)) + 
      R*((39.47841760435743*(-10 + R)*Z*Cos(4*Pi*(-10 + R))*Cos(4*Pi*Z)*
            Sin(4*P)*Sin(4*Pi*t))/
          (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
         (6.283185307179586*Power(-10 + R,2)*Z*Cos(4*Pi*Z)*Sin(4*P)*
            Sin(4*Pi*(-10 + R))*Sin(4*Pi*t))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
         (3.141592653589793*Z*Cos(4*Pi*Z)*Sin(4*P)*Sin(4*Pi*(-10 + R))*
            Sin(4*Pi*t))/(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
         (251.32741228718345*Z*Cos(4*P)*Cos(4*Pi*(-10 + R))*Sin(4*Pi*t)*
            Sin(4*Pi*Z))/(R*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
         (6.283185307179586*(-10 + R)*Power(Z,2)*Cos(4*Pi*(-10 + R))*
            Sin(4*P)*Sin(4*Pi*t)*Sin(4*Pi*Z))/
          Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2) + 
         (40.*(-10 + R)*Z*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*
            Sin(4*Pi*Z))/
          (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),2)) + 
         (20.*Z*Cos(4*P)*Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z))/
          (Power(R,2)*(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
         4*Power(Pi,2)*(1 - (1.*Power(Z,2))/
             (400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))*Sin(4*P)*
          Sin(4*Pi*(-10 + R))*Sin(4*Pi*t)*Sin(4*Pi*Z)))/R)
; }};
