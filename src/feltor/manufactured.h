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
    return 1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/R + 
  (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct Ue{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 52.63789013914324*R*Sin(2*Pi*R))*
    Sin(2*Pi*t)*Sin(2*Pi*Z))/
  (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))
; }};
struct Ui{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
      10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
  (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
    R*(0.4052847345693511 + (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))
; }};
struct we{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*t)*((beta*(1 + Sqrt(-mue))*Sin(2*Pi*R))/mue + 
      (12.566370614359172*Cos(2*Pi*R) - 157.91367041742973*R*Sin(2*Pi*R))/
       (R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))*Sin(2*Pi*Z))/
  (3.*Sqrt(-mue))
; }};
struct Wi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*t)*((beta*(1 + Sqrt(-mue))*Sin(2*Pi*R))/
       (Sqrt(-mue)*mui) + (-2.5464790894703255*Cos(2*Pi*R) + 
         31.999999999999996*R*Sin(2*Pi*R))/
       (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
          Sin(Pi*Z) + R*(0.4052847345693511 + 
            (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
             Sin(Pi*Z))))*Sin(2*Pi*Z))/3.
; }};
struct Phie{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*(-0.942477796076938*mui*taui*Cos(3*Pi*R) + 
      R*(0.20000000000000004 + 17.765287921960844*mui*taui)*Sin(3*Pi*R))*
    Sin(3*Pi*t)*Sin(3*Pi*Z))/R
; }};
struct Phii{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*t)*(40*Sin(3*Pi*R)*Sin(3*Pi*Z) - 
      (mui*Sin(3*P)*Sin(3*Pi*t)*
         (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
            Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
              R*(-0.20000000000000004 - 17.765287921960844*mui*taui)*
               Sin(3*Pi*R),2) + 
           28034.10888465587*Power((0.005628954646796544*mui*taui + 
                 Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
               Cos(3*Pi*R) + 0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
            Power(Sin(3*Pi*Z),2)))/
       (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))/200.
; }};
struct GammaPhie{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(3*P)*Sin(3*Pi*(-10 + R))*Sin(3*Pi*t)*Sin(3*Pi*Z))/5.
; }};
struct GammaNi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.25*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)
; }};
struct A{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (beta*(1 + Sqrt(-mue))*Sin(2*P)*Sin(2*Pi*R)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
  (3.*Sqrt(-mue))
; }};
struct SNe{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1.5707963267948966*Cos(Pi*t)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*Z) + 
  (FELTORPARALLEL*(0. - (6.283185307179586*(-10 + R)*Cos(2*Pi*Z)*Sin(2*P)*
          (4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
       (3.141592653589793*(-10 + R)*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
          (4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
       (3.141592653589793*Z*Cos(Pi*R)*Sin(P)*Sin(2*P)*
          (4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(Pi*Z)*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
       (20.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
          (4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(Pi*Z)*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
             Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(-mue)*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
       (1.5707963267948966*(-10 + R)*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
          Sin(Pi*(-10 + R))*(4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
       (1.5707963267948966*Z*Cos(Pi*(-10 + R))*Sin(P)*Sin(2*P)*
          (4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(Pi*Z)*Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
       (10.*Cos(P)*Sin(2*P)*Sin(Pi*(-10 + R))*
          (4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(Pi*Z)*Sin(2*Pi*Z))/
        (Sqrt(-mue)*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
       (1.*Z*Sin(2*P)*(-330.73361792319804*R*Cos(2*Pi*R) - 
            78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/
        (Sqrt(-mue)*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
       (40*Cos(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
          Sin(2*Pi*Z))/
        (Sqrt(-mue)*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/R + 
  FELTORPERP*(-((nuperp*(0. + 1.5707963267948966*Cos(Pi*(-10 + R))*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z) - 
            9.869604401089358*R*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z)\
))/R) + (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((0.15000000000000002*R*(-20. + 2.*R)*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.1*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (0.8882643960980423*Cos(3*Pi*Z)*Sin(3*P)*
              (3*Pi*R*(-0.2122065907891938 - 
                   18.849555921538755*mui*taui)*Cos(3*Pi*R) - 
                9.42477796076938*mui*taui*Sin(3*Pi*R) + 
                (-0.2122065907891938 - 18.849555921538755*mui*taui)*
                 Sin(3*Pi*R))*Sin(3*Pi*t))/
            Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
           (0.44413219804902115*(-20. + 2.*R)*Cos(3*Pi*Z)*Sin(3*P)*
              (1.*mui*taui*Cos(3*Pi*R) + 
                R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*
                 Sin(3*Pi*R))*Sin(3*Pi*t))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (0.6579736267392905*beta*(1 + Sqrt(-mue))*Cos(Pi*R)*
              Cos(2*Pi*Z)*Sin(P)*Power(Sin(2*P),2)*Sin(2*Pi*R)*
              (4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
              Power(Sin(2*Pi*t),2)*Sin(Pi*Z)*Sin(2*Pi*Z))/
            (mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
           (beta*(1 + Sqrt(-mue))*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              (-330.73361792319804*R*Cos(2*Pi*R) - 
                78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*R)*
              Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
            (15.*mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
           (2*beta*(1 + Sqrt(-mue))*Power(Pi,2)*Cos(2*Pi*R)*Cos(2*Pi*Z)*
              Power(Sin(2*P),2)*
              (4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
           (beta*(1 + Sqrt(-mue))*Pi*(-20. + 2.*R)*Cos(2*Pi*Z)*
              Power(Sin(2*P),2)*Sin(2*Pi*R)*
              (4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (30.*mue*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
               1.5)*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) + 
        1.5707963267948966*R*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*Sin(Pi*Z)*
         ((-0.1*R*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (0.8882643960980423*Cos(3*Pi*Z)*Sin(3*P)*
              (1.*mui*taui*Cos(3*Pi*R) + 
                R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*
                 Sin(3*Pi*R))*Sin(3*Pi*t))/
            Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
           (beta*(1 + Sqrt(-mue))*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              Sin(2*Pi*R)*(4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) + 
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((-0.1*R*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (0.8882643960980423*Cos(3*Pi*Z)*Sin(3*P)*
              (1.*mui*taui*Cos(3*Pi*R) + 
                R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*
                 Sin(3*Pi*R))*Sin(3*Pi*t))/
            Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
           (beta*(1 + Sqrt(-mue))*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              Sin(2*Pi*R)*(4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) + 
        R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
         ((-3.*taue*Z*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.2*taue*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (3*Pi*Cos(3*Pi*Z)*Sin(3*P)*
              ((0.09424777960769379*mui*taui + 
                   Power(R,2)*
                    (0.1884955592153876 + 16.7433894073619*mui*taui))*
                 Cos(3*Pi*R) + 0.8882643960980426*mui*R*taui*Sin(3*Pi*R))*
              Sin(3*Pi*t))/
            (R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (2*beta*(1 + Sqrt(-mue))*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
           (0.10471975511965977*beta*(1 + Sqrt(-mue))*Cos(Pi*Z)*Sin(P)*
              Power(Sin(2*P),2)*Sin(Pi*R)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
            (mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
           (0.03333333333333333*beta*(1 + Sqrt(-mue))*Z*Power(Sin(2*P),2)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Power(Sin(2*Pi*Z),2))/
            (mue*R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)*
              (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
           ((-34818.239691125345*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
                 Power(0.07957747154594767*Cos(2*Pi*R) - 
                   1.*R*Sin(2*Pi*R),2)*Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
               (Power(R,2)*Power(2. + 
                   1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
              (17409.119845562673*Cos(Pi*Z)*Sin(P)*Power(Sin(2*P),2)*
                 Sin(Pi*R)*Power(0.07957747154594767*Cos(2*Pi*R) - 
                   1.*R*Sin(2*Pi*R),2)*Sin(Pi*t)*Power(Sin(2*Pi*t),2)*
                 Power(Sin(2*Pi*Z),2))/
               (Power(R,2)*Power(2. + 
                   1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)))/
            (10.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (0.1*Z*(taue - (2770.7474783005127*Power(Sin(2*P),2)*
                   Power(0.07957747154594767*Cos(2*Pi*R) - 
                     1.*R*Sin(2*Pi*R),2)*Power(Sin(2*Pi*t),2)*
                   Power(Sin(2*Pi*Z),2))/
                 (Power(R,2)*Power(2. + 
                     1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (1.*Z*Sin(3*P)*((0.09424777960769379*mui*taui + 
                   Power(R,2)*
                    (0.1884955592153876 + 16.7433894073619*mui*taui))*
                 Cos(3*Pi*R) + 0.8882643960980426*mui*R*taui*Sin(3*Pi*R))*
              Sin(3*Pi*t)*Sin(3*Pi*Z))/
            (R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5))) + 
        1.5707963267948966*R*Cos(Pi*Z)*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
         ((taue*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (beta*(1 + Sqrt(-mue))*Power(Sin(2*P),2)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (4.1887902047863905*Cos(2*Pi*R) - 
                52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Power(Sin(2*Pi*Z),2))/
            (30.*mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
              (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
           (taue - (2770.7474783005127*Power(Sin(2*P),2)*
                 Power(0.07957747154594767*Cos(2*Pi*R) - 
                   1.*R*Sin(2*Pi*R),2)*Power(Sin(2*Pi*t),2)*
                 Power(Sin(2*Pi*Z),2))/
               (Power(R,2)*Power(2. + 
                   1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)))/
            (10.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (Sin(3*P)*((0.09424777960769379*mui*taui + 
                   Power(R,2)*
                    (0.1884955592153876 + 16.7433894073619*mui*taui))*
                 Cos(3*Pi*R) + 0.8882643960980426*mui*R*taui*Sin(3*Pi*R))*
              Sin(3*Pi*t)*Sin(3*Pi*Z))/
            (R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))/R)
; }};
struct SNi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (-1.2337005501361697*mui*taui*Cos(Pi*R)*Cos(Pi*t)*Sin(P)*Sin(Pi*Z))/R + 
  Pi*(0.25 + 2.4674011002723395*mui*taui)*Cos(Pi*t)*Sin(P)*Sin(Pi*R)*
   Sin(Pi*Z) + (FELTORPARALLEL*
     (0. - (6.283185307179586*(-10 + R)*R*Cos(2*Pi*Z)*Sin(2*P)*
          (-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z)))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)))) + 
       (1.*(-10 + R)*R*Sin(2*P)*
          (-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*
          (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
            Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*R)*Sin(Pi*t))*Sin(2*Pi*t)*
          (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)),2)) - 
       (20*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (-0.15915494309189535*mui*taui*Cos(P)*Cos(Pi*R)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
             Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
          (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)),2)) - 
       (1.*R*Z*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (0.4052847345693511 + 
            Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z) + 
            0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
            (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*
          (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)),2)) - 
       (1.*(-10 + R)*R*Sin(2*P)*
          (-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*
          ((-1.2337005501361697*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
               Sin(Pi*t))/R + 
            Pi*(0.25 + 2.4674011002723395*mui*taui)*Cos(Pi*Z)*Sin(P)*
             Sin(Pi*R)*Sin(Pi*t))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)))) + 
       (20*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          ((-0.39269908169872414*mui*taui*Cos(P)*Cos(Pi*R)*Sin(Pi*t)*
               Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Cos(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)))) + 
       (1.*R*Z*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          ((0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
               Sin(Pi*Z))/Power(R,2) + 
            Pi*(0.25 + 2.4674011002723395*mui*taui)*Cos(Pi*R)*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z) + 
            (1.2337005501361697*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
               Sin(Pi*Z))/R)*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)))) + 
       (1.*R*Z*Sin(2*P)*(67.02064327658225*R*Cos(2*Pi*R) + 
            16.*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)))) + 
       (40*Cos(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)))) + 
       (1.*Z*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
            10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
          (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/R + 
            (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
             Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
        (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
          (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z))))))/R + 
  FELTORPERP*(-((nuperp*((0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z))/Power(R,2) + 
            Pi*(0.25 + 2.4674011002723395*mui*taui)*Cos(Pi*R)*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z) + 
            (1.2337005501361697*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
               Sin(Pi*Z))/R + R*
             ((3.875784585037477*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                  Sin(Pi*Z))/R - 
               Power(Pi,2)*(0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
                Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)) + 
            R*((-0.7853981633974483*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                  Sin(Pi*Z))/Power(R,3) + 
               (3.875784585037477*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                  Sin(Pi*Z))/R - 
               (2.4674011002723395*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                  Sin(Pi*Z))/Power(R,2) - 
               Power(Pi,2)*(0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
                Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))/R) + 
     (R*((0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z))/Power(R,2) + 
           Pi*(0.25 + 2.4674011002723395*mui*taui)*Cos(Pi*R)*Sin(P)*
            Sin(Pi*t)*Sin(Pi*Z) + 
           (1.2337005501361697*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z))/R)*((-0.1*R*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (beta*(1 + Sqrt(-mue))*Pi*R*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              Sin(2*Pi*R)*(-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) - 
           (R*Sin(3*P)*Sin(3*Pi*t)*
              (120*Pi*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
                (mui*Cos(3*Pi*Z)*Sin(3*P)*
                   (528430.5031318273*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
                     54*Power(Pi,3)*Power(R,2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                   Sin(3*Pi*t)*Sin(3*Pi*Z))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) + 
                (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2))))/
            (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
        (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
            Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
         ((-0.1*R*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (beta*(1 + Sqrt(-mue))*Pi*R*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              Sin(2*Pi*R)*(-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) - 
           (R*Sin(3*P)*Sin(3*Pi*t)*
              (120*Pi*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
                (mui*Cos(3*Pi*Z)*Sin(3*P)*
                   (528430.5031318273*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
                     54*Power(Pi,3)*Power(R,2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                   Sin(3*Pi*t)*Sin(3*Pi*Z))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) + 
                (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2))))/
            (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
        R*(1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
            Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
         ((0.15000000000000002*R*(-20. + 2.*R)*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.1*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (beta*(1 + Sqrt(-mue))*Pi*R*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              Sin(2*Pi*R)*(-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              (0.4052847345693511 + 
                Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                 Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),2)) + 
           (beta*(1 + Sqrt(-mue))*Pi*R*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              Sin(2*Pi*R)*(67.02064327658225*R*Cos(2*Pi*R) + 
                16.*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) + 
           (2*beta*(1 + Sqrt(-mue))*Power(Pi,2)*R*Cos(2*Pi*R)*Cos(2*Pi*Z)*
              Power(Sin(2*P),2)*
              (-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) - 
           (beta*(1 + Sqrt(-mue))*Pi*R*(-20. + 2.*R)*Cos(2*Pi*Z)*
              Power(Sin(2*P),2)*Sin(2*Pi*R)*
              (-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (30.*Sqrt(-mue)*Power(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2),1.5)*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) + 
           (beta*(1 + Sqrt(-mue))*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              Sin(2*Pi*R)*(-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) + 
           (R*(-20. + 2.*R)*Sin(3*P)*Sin(3*Pi*t)*
              (120*Pi*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
                (mui*Cos(3*Pi*Z)*Sin(3*P)*
                   (528430.5031318273*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                       Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) \
- 54*Power(Pi,3)*Power(R,2)*Power(0.942477796076938*mui*taui*
                        Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                   Sin(3*Pi*t)*Sin(3*Pi*Z))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) + 
                (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2))))/
            (4000.*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)) \
- (Sin(3*P)*Sin(3*Pi*t)*(120*Pi*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
                (mui*Cos(3*Pi*Z)*Sin(3*P)*
                   (528430.5031318273*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                       Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) \
- 54*Power(Pi,3)*Power(R,2)*Power(0.942477796076938*mui*taui*
                        Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                   Sin(3*Pi*t)*Sin(3*Pi*Z))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) + 
                (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2))))/
            (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (R*Sin(3*P)*Sin(3*Pi*t)*
              (360*Power(Pi,2)*Cos(3*Pi*R)*Cos(3*Pi*Z) + 
                (mui*(-20. + 2.*R)*Cos(3*Pi*Z)*Sin(3*P)*
                   (528430.5031318273*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
                     54*Power(Pi,3)*Power(R,2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                   Sin(3*Pi*t)*Sin(3*Pi*Z))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2)) + 
                (2*mui*Cos(3*Pi*Z)*Sin(3*P)*
                   (528430.5031318273*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
                     54*Power(Pi,3)*Power(R,2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                   Sin(3*Pi*t)*Sin(3*Pi*Z))/
                 (Power(R,3)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) - 
                (mui*Cos(3*Pi*Z)*Sin(3*P)*
                   (-108*Power(Pi,3)*Power(R,2)*
                      (3*Pi*R*
                        (-0.20000000000000004 - 
                       17.765287921960844*mui*taui)*Cos(3*Pi*R) - 
                        8.882643960980424*mui*taui*Sin(3*Pi*R) + 
                        (-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R))*
                      (0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R)) - 
                     108*Power(Pi,3)*R*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     1.0568610062636547e6*
                      ((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                      (0.5000000000000001*mui*R*taui*Cos(3*Pi*R) + 
                        2*R*(0.011257909293593089 + 1.*mui*taui)*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*taui*Sin(3*Pi*R) - 
                        3*Pi*
                         (0.005628954646796544*mui*taui + 
                        Power(R,2)*(0.011257909293593089 + 1.*mui*taui)\
)*Sin(3*Pi*R)))*Sin(3*Pi*t)*Sin(3*Pi*Z))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) - 
                (4.*mui*(-20. + 2.*R)*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),3)) - 
                (4.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,3)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2)) + 
                (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (18*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      (3*Pi*R*
                        (-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Cos(3*Pi*R) - 
                        8.882643960980424*mui*taui*Sin(3*Pi*R) + 
                        (-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R))*
                      (0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                         17.765287921960844*mui*taui)*Sin(3*Pi*R)) + 
                     18*Power(Pi,2)*R*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     56068.21776931174*
                      ((0.005628954646796544*mui*taui + 
                        Power(R,2)*(0.011257909293593089 + 1.*mui*taui)\
)*Cos(3*Pi*R) + 0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                      (0.5000000000000001*mui*R*taui*Cos(3*Pi*R) + 
                        2*R*(0.011257909293593089 + 1.*mui*taui)*
                         Cos(3*Pi*R) + 
                        0.05305164769729846*mui*taui*Sin(3*Pi*R) - 
                        3*Pi*(0.005628954646796544*mui*taui + 
                         Power(R,2)*(0.011257909293593089 + 1.*mui*taui)\
)*Sin(3*Pi*R))*Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2))))/
            (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
        R*(1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
            Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
         ((-3.*taui*Z*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5) - 
           (0.2*taui*Z)/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (2*beta*(1 + Sqrt(-mue))*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Sin(2*Pi*Z))/
            (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) + 
           (beta*(1 + Sqrt(-mue))*Power(Sin(2*P),2)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*
              (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
                Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*
                 Sin(P)*Sin(Pi*R)*Sin(Pi*t))*Power(Sin(2*Pi*t),2)*
              Power(Sin(2*Pi*Z),2))/
            (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),2)) + 
           (0.03333333333333333*beta*(1 + Sqrt(-mue))*Z*Power(Sin(2*P),2)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Power(Sin(2*Pi*Z),2))/
            (Sqrt(-mue)*Power(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2),1.5)*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) - 
           ((4*mui*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
                 Power(0.8488263631567752*Cos(2*Pi*R) - 
                   10.666666666666666*R*Sin(2*Pi*R),2)*
                 Power(Sin(2*Pi*t),2)*Sin(2*Pi*Z))/
               Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                  Sin(Pi*t)*Sin(Pi*Z) + 
                 R*(0.4052847345693511 + 
                    (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                     Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
              (2*mui*Power(Sin(2*P),2)*
                 Power(0.8488263631567752*Cos(2*Pi*R) - 
                   10.666666666666666*R*Sin(2*Pi*R),2)*
                 (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
                   Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*
                    Sin(P)*Sin(Pi*R)*Sin(Pi*t))*Power(Sin(2*Pi*t),2)*
                 Power(Sin(2*Pi*Z),2))/
               Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                  Sin(Pi*t)*Sin(Pi*Z) + 
                 R*(0.4052847345693511 + 
                    (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                     Sin(Pi*t)*Sin(Pi*Z)),3))/
            (10.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (0.1*Z*(taui + (mui*Power(Sin(2*P),2)*
                   Power(0.8488263631567752*Cos(2*Pi*R) - 
                     10.666666666666666*R*Sin(2*Pi*R),2)*
                   Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
                 Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                    Sin(Pi*t)*Sin(Pi*Z) + 
                   R*(0.4052847345693511 + 
                      (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                       Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
           (R*Sin(3*P)*Sin(3*Pi*t)*
              (360*Power(Pi,2)*Cos(3*Pi*R)*Cos(3*Pi*Z) + 
                (mui*(-20. + 2.*R)*Sin(3*P)*Sin(3*Pi*t)*
                   (528430.5031318273*Cos(3*Pi*Z)*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                       Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Sin(3*Pi*Z) - 
                     54*Power(Pi,3)*Power(R,2)*Cos(3*Pi*Z)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2)*
                      Sin(3*Pi*Z)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2)) + 
                (2*mui*Sin(3*P)*Sin(3*Pi*t)*
                   (528430.5031318273*Cos(3*Pi*Z)*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                       Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Sin(3*Pi*Z) - 
                     54*Power(Pi,3)*Power(R,2)*Cos(3*Pi*Z)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2)*
                      Sin(3*Pi*Z)))/
                 (Power(R,3)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) - 
                (mui*Sin(3*P)*Sin(3*Pi*t)*
                   (-108*Power(Pi,3)*R*Cos(3*Pi*Z)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                       17.765287921960844*mui*taui)*Sin(3*Pi*R),2)*
                      Sin(3*Pi*Z) - 
                     9.960680319430241e6*Power(R,2)*Cos(3*Pi*Z)*
                      (R*(0.011257909293593089 + 1.*mui*taui)*
                        Cos(3*Pi*R) + 
                        (0.0011945012753036852 + 
                        0.15915494309189537*mui*taui)*Sin(3*Pi*R))*
                      (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
                        R*(0.011257909293593089 + 1.*mui*taui)*
                        Sin(3*Pi*R))*Sin(3*Pi*Z) - 
                     3.1705830187909645e6*Cos(3*Pi*Z)*
                      ((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                      (R*(-0.007505272862395392 - 
                        0.8333333333333334*mui*taui)*Cos(3*Pi*R) + 
                        (-6.938893903907228e-18*mui*taui + 
                        Power(R,2)*
                        (0.03536776513153231 + 
                        3.141592653589793*mui*taui))*Sin(3*Pi*R))*
                      Sin(3*Pi*Z)))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) - 
                (4.*mui*(-20. + 2.*R)*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),3)) - 
                (4.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,3)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2)) + 
                (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                   (18*Power(Pi,2)*R*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     528430.5031318273*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      (R*(0.011257909293593089 + 1.*mui*taui)*
                        Cos(3*Pi*R) + 
                        (0.0011945012753036852 + 
                        0.15915494309189537*mui*taui)*Sin(3*Pi*R))*
                      (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
                        R*(0.011257909293593089 + 1.*mui*taui)*
                         Sin(3*Pi*R)) - 
                     168204.65330793522*
                      ((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                      (R*(-0.007505272862395392 - 
                        0.8333333333333334*mui*taui)*Cos(3*Pi*R) + 
                        (-6.938893903907228e-18*mui*taui + 
                        Power(R,2)*
                        (0.03536776513153231 + 
                        3.141592653589793*mui*taui))*Sin(3*Pi*R))*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2))))/
            (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
           (0.0005*R*Z*Sin(3*P)*Sin(3*Pi*t)*
              (120*Pi*Cos(3*Pi*R)*Sin(3*Pi*Z) + 
                (mui*(-20. + 2.*R)*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2)) + 
                (2*mui*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,3)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) - 
                (mui*Sin(3*P)*Sin(3*Pi*t)*
                   (18*Power(Pi,2)*R*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     528430.5031318273*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      (R*(0.011257909293593089 + 1.*mui*taui)*
                        Cos(3*Pi*R) + 
                        (0.0011945012753036852 + 
                        0.15915494309189537*mui*taui)*Sin(3*Pi*R))*
                      (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
                        R*(0.011257909293593089 + 1.*mui*taui)*
                         Sin(3*Pi*R)) - 
                     168204.65330793522*
                      ((0.005628954646796544*mui*taui + 
                        Power(R,2)*(0.011257909293593089 + 1.*mui*taui)\
)*Cos(3*Pi*R) + 0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                      (R*(-0.007505272862395392 - 
                        0.8333333333333334*mui*taui)*Cos(3*Pi*R) + 
                        (-6.938893903907228e-18*mui*taui + 
                         Power(R,2)*
                         (0.03536776513153231 + 
                         3.141592653589793*mui*taui))*Sin(3*Pi*R))*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2)))))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)) + 
        R*((-1.2337005501361697*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
              Sin(Pi*t))/R + Pi*(0.25 + 2.4674011002723395*mui*taui)*
            Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
         ((taui*(-50. + 1.*R - 0.1*Power(Z,2)))/
            Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
           (beta*(1 + Sqrt(-mue))*Power(Sin(2*P),2)*
              (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
              (-0.8488263631567752*Cos(2*Pi*R) + 
                10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
              Power(Sin(2*Pi*Z),2))/
            (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2))*
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) - 
           (taui + (mui*Power(Sin(2*P),2)*
                 Power(0.8488263631567752*Cos(2*Pi*R) - 
                   10.666666666666666*R*Sin(2*Pi*R),2)*
                 Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
               Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                  Sin(Pi*t)*Sin(Pi*Z) + 
                 R*(0.4052847345693511 + 
                    (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                     Sin(Pi*t)*Sin(Pi*Z)),2))/
            (10.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
           (R*Sin(3*P)*Sin(3*Pi*t)*
              (120*Pi*Cos(3*Pi*R)*Sin(3*Pi*Z) + 
                (mui*(-20. + 2.*R)*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2),2)) + 
                (2*mui*Sin(3*P)*Sin(3*Pi*t)*
                   (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     28034.10888465587*
                      Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                        0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,3)*(500. - 20.*R + 1.*Power(R,2) + 
                     1.*Power(Z,2))) - 
                (mui*Sin(3*P)*Sin(3*Pi*t)*
                   (18*Power(Pi,2)*R*Power(Cos(3*Pi*Z),2)*
                      Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                        R*(-0.20000000000000004 - 
                         17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                     528430.5031318273*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                      (R*(0.011257909293593089 + 1.*mui*taui)*
                         Cos(3*Pi*R) + 
                        (0.0011945012753036852 + 
                         0.15915494309189537*mui*taui)*Sin(3*Pi*R))*
                      (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
                        R*(0.011257909293593089 + 1.*mui*taui)*Sin(3*Pi*R)\
) - 168204.65330793522*((0.005628954646796544*mui*taui + 
                         Power(R,2)*(0.011257909293593089 + 1.*mui*taui)\
)*Cos(3*Pi*R) + 0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                      (R*(-0.007505272862395392 - 
                         0.8333333333333334*mui*taui)*Cos(3*Pi*R) + 
                        (-6.938893903907228e-18*mui*taui + 
                         Power(R,2)*
                         (0.03536776513153231 + 
                         3.141592653589793*mui*taui))*Sin(3*Pi*R))*
                      Power(Sin(3*Pi*Z),2)))/
                 (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))\
))/(2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))/R)
; }};
struct SWe{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (-1.0471975511965976*Cos(Pi*t)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
     (12.566370614359172*Cos(2*Pi*R) - 157.91367041742973*R*Sin(2*Pi*R))*
     Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
   (Sqrt(-mue)*R*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
  (2*Pi*Cos(2*Pi*t)*Sin(2*P)*((beta*(1 + Sqrt(-mue))*Sin(2*Pi*R))/mue + 
       (12.566370614359172*Cos(2*Pi*R) - 157.91367041742973*R*Sin(2*Pi*R))/
        (R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))*Sin(2*Pi*Z))/
   (3.*Sqrt(-mue)) + FELTORPARALLEL*
   ((taue*Sin(Pi*t)*((15.707963267948966 - 1.5707963267948966*R)*R*
           Cos(Pi*Z)*Sin(P)*Sin(Pi*R) + 
          (1.5707963267948966*R*Z*Cos(Pi*R)*Sin(P) + 10.*Cos(P)*Sin(Pi*R))*
           Sin(Pi*Z)))/
      (mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) - 
     (1.*(-10 + R)*Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
          52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
           (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
          (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 
               1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
      (Sqrt(-mue)*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
     (20*Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
          52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        ((-1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(Pi*Z)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 
               1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
          (2*Cos(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
      (Sqrt(-mue)*Power(R,2)*Sqrt(400 + 1.*Power(-10 + R,2) + 
          1.*Power(Z,2))*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
     (1.*Z*Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
          52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        ((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(Pi*Z)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 
               1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
          (Sin(2*P)*(-330.73361792319804*R*Cos(2*Pi*R) - 
               78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
          (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*Power(R,2)*
             (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
      (Sqrt(-mue)*R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
     (nuparallele*(0. + (1.*Z*
             ((-1.*(-10 + R)*
                  ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (20*((-1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Cos(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))\
)/(R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
               (1.*Z*((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20*((-1.*(-10 + R)*
                  ((-6.283185307179586*Cos(P)*Cos(2*Pi*Z)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
+ (6.283185307179586*Cos(P)*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
                       Power(Sin(Pi*R),2)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*Sin(Pi*Z)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (6.283185307179586*Cos(2*P)*Cos(Pi*Z)*Sin(P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (3.141592653589793*Cos(P)*Cos(Pi*Z)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (1.*Z*((6.283185307179586*Cos(P)*Cos(Pi*R)*Sin(P)*
                       Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*
                       Power(Sin(Pi*Z),2)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                       78.95683520871486*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (6.283185307179586*Cos(2*P)*Cos(Pi*R)*Sin(P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (3.141592653589793*Cos(P)*Cos(Pi*R)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Cos(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                       78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (2*Cos(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))\
)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (20*((2.*Power(Cos(P),2)*Sin(2*P)*Power(Sin(Pi*R),2)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*
                       Power(Sin(Pi*Z),2)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (4.*Cos(P)*Cos(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (1.*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (4*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (1.*(-10 + R)*R*((20*
                  ((-6.283185307179586*Cos(P)*Cos(2*Pi*Z)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
+ (6.283185307179586*Cos(P)*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
                       Power(Sin(Pi*R),2)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*Sin(Pi*Z)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (6.283185307179586*Cos(2*P)*Cos(Pi*Z)*Sin(P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (3.141592653589793*Cos(P)*Cos(Pi*Z)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
               (1.*Z*((-19.739208802178716*Cos(Pi*R)*Cos(2*Pi*Z)*
                       Sin(P)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                       78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
+ (19.739208802178716*Cos(Pi*R)*Cos(Pi*Z)*Power(Sin(P),2)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*Sin(Pi*Z)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
                       Sin(Pi*R)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                       78.95683520871486*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (9.869604401089358*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                       Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (1.*(-10 + R)*Z*
                  ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
               (20.*Z*((-1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Cos(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))\
)/(R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) - 
               (1.*Power(Z,2)*
                  ((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                       78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))\
)/Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) + 
               (1.*((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                       52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                       78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))\
)/Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
               (1.*(-10 + R)*
                  ((-39.47841760435743*Cos(Pi*Z)*Cos(2*Pi*Z)*Sin(P)*
                       Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (19.739208802178716*Power(Cos(Pi*Z),2)*
                       Power(Sin(P),2)*Sin(2*P)*Power(Sin(Pi*R),2)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) + 
                    (9.869604401089358*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (4*Power(Pi,2)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (1.*R*Z*((-1.*(-10 + R)*
                  ((-19.739208802178716*Cos(Pi*R)*Cos(2*Pi*Z)*Sin(P)*
                       Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
+ (19.739208802178716*Cos(Pi*R)*Cos(Pi*Z)*Power(Sin(P),2)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*Sin(Pi*Z)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
                       Sin(Pi*R)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (9.869604401089358*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                       Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*
                       Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (1.*Power(-10 + R,2)*
                  ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
               (1.*((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (20*((6.283185307179586*Cos(P)*Cos(Pi*R)*Sin(P)*
                       Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*
                       Power(Sin(Pi*Z),2)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (6.283185307179586*Cos(2*P)*Cos(Pi*R)*Sin(P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (3.141592653589793*Cos(P)*Cos(Pi*R)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Cos(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (2*Cos(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
               (20.*(-10 + R)*
                  ((-1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Cos(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) \
- (20*((-1.*Cos(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (2*Cos(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (Power(R,2)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2))) - 
               (1.*(-10 + R)*Z*
                  ((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) \
- (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) + 
               (1.*Z*((19.739208802178716*Power(Cos(Pi*R),2)*
                       Power(Sin(P),2)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*
                       Power(Sin(Pi*t),2)*Sin(2*Pi*t)*
                       Power(Sin(Pi*Z),2)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
                    (6.283185307179586*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (6.283185307179586*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
                    (9.869604401089358*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*
                       Sin(2*Pi*t)*Sin(Pi*Z)*Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       Power(2. + 
                        1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
                    (2*Sin(2*P)*
                       (-330.73361792319804*R*Cos(2*Pi*R) - 
                        78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,2)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
                    (2*Sin(2*P)*
                       (4.1887902047863905*Cos(2*Pi*R) - 
                        52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*Power(R,3)*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
                    (Sin(2*P)*
                       (-826.834044807995*Cos(2*Pi*R) + 
                        2078.060608725385*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (Sqrt(-mue)*R*
                       (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
      (R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     (Sin(3*Pi*t)*(R*Sin(3*Pi*R)*
           (R*(18.849555921538762 + 1674.3389407361901*mui*taui + 
                R*(-1.884955592153876 - 167.433894073619*mui*taui))*
              Cos(3*Pi*Z)*Sin(3*P) + 
             ((12. + 1065.9172753176506*mui*taui)*Cos(3*P) + 
                8.882643960980428*mui*taui*Z*Sin(3*P))*Sin(3*Pi*Z)) + 
          Cos(3*Pi*R)*(mui*R*(-88.82643960980423 + 8.882643960980422*R)*
              taui*Cos(3*Pi*Z)*Sin(3*P) + 
             (-56.54866776461628*mui*taui*Cos(3*P) + 
                (0.9424777960769379*mui*taui + 
                   Power(R,2)*
                    (1.884955592153876 + 167.433894073619*mui*taui))*Z*
                 Sin(3*P))*Sin(3*Pi*Z))))/
      (mue*Power(R,2)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
  FELTORPERP*(0. - (0.1*taue*Z*Sin(2*P)*
        (4.1887902047863905*Cos(2*Pi*R) - 52.63789013914324*R*Sin(2*Pi*R))*
        Sin(2*Pi*t)*Sin(2*Pi*Z))/
      (Sqrt(-mue)*R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)*
        (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
     (0.8882643960980423*Cos(3*Pi*Z)*Sin(2*P)*Sin(3*P)*
        (4.1887902047863905*Cos(2*Pi*R) - 52.63789013914324*R*Sin(2*Pi*R))*
        (1.*mui*taui*Cos(3*Pi*R) + 
          R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*Sin(3*Pi*R)\
)*Sin(2*Pi*t)*Sin(3*Pi*t)*Sin(2*Pi*Z))/
      (Sqrt(-mue)*Power(R,2)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
          1.*Power(Z,2))*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
     (0.32898681336964525*beta*(1. + Sqrt(-mue))*taue*Sin(P)*Sin(2*P)*
        Sin(Pi*t)*Sin(2*Pi*t)*(1.*R*Cos(Pi*R)*Cos(2*Pi*Z)*Sin(2*Pi*R)*
           Sin(Pi*Z) + Cos(Pi*Z)*Sin(Pi*R)*
           (-1.*R*Cos(2*Pi*R) - 0.15915494309189535*Sin(2*Pi*R))*Sin(2*Pi*Z)\
))/(Sqrt(-mue)*mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) + 
     ((33.0733617923198*taue*Cos(2*Pi*Z)*Sin(2*P)*
           (-0.07957747154594767*Cos(2*Pi*R) + 1.*R*Sin(2*Pi*R))*
           Sin(2*Pi*t))/
         (Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
        (5.263789013914324*taue*Z*Sin(2*P)*
           (-0.07957747154594767*Cos(2*Pi*R) + 1.*R*Sin(2*Pi*R))*
           Sin(2*Pi*t)*Sin(2*Pi*Z))/
         (Sqrt(-mue)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
            1.5)))/(R*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))) \
+ ((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
           (4.1887902047863905*Cos(2*Pi*R) - 
             52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
           Sin(Pi*Z)*Sin(2*Pi*Z))/
         (Sqrt(-mue)*R*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),
            2)) + (Sin(2*P)*(-330.73361792319804*R*Cos(2*Pi*R) - 
             78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
         (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
        (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
             52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
         (Sqrt(-mue)*Power(R,2)*
           (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))*
      ((-0.1*R*taue*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
        (0.8882643960980423*Cos(3*Pi*Z)*Sin(3*P)*
           (1.*mui*taui*Cos(3*Pi*R) + 
             R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*
              Sin(3*Pi*R))*Sin(3*Pi*t))/
         Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) - 
        (beta*(1 + Sqrt(-mue))*Pi*Cos(2*Pi*Z)*Power(Sin(2*P),2)*Sin(2*Pi*R)*
           (4.1887902047863905*Cos(2*Pi*R) - 
             52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
           Sin(2*Pi*Z))/
         (15.*mue*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
           (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) + 
     (eta*(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z))*
        (-((Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                 52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
             (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) \
+ (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)))))/mue - 
     (nuperp*((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(Pi*Z)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 
               1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
          (Sin(2*P)*(-330.73361792319804*R*Cos(2*Pi*R) - 
               78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
          (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*Power(R,2)*
             (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
          R*((-39.47841760435743*Cos(Pi*Z)*Cos(2*Pi*Z)*Sin(P)*Sin(2*P)*
                Sin(Pi*R)*(4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t))/
              (Sqrt(-mue)*R*Power(2. + 
                  1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
             (19.739208802178716*Power(Cos(Pi*Z),2)*Power(Sin(P),2)*
                Sin(2*P)*Power(Sin(Pi*R),2)*
                (4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(Pi*t),2)*
                Sin(2*Pi*t)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*R*Power(2. + 
                  1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) + 
             (9.869604401089358*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                (4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
                Sin(Pi*Z)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*R*Power(2. + 
                  1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
             (4*Power(Pi,2)*Sin(2*P)*
                (4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)\
)/(Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) + 
          R*((19.739208802178716*Power(Cos(Pi*R),2)*Power(Sin(P),2)*
                Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(Pi*t),2)*
                Sin(2*Pi*t)*Power(Sin(Pi*Z),2)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*R*Power(2. + 
                  1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) - 
             (6.283185307179586*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                (-330.73361792319804*R*Cos(2*Pi*R) - 
                  78.95683520871486*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
                Sin(Pi*Z)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*R*Power(2. + 
                  1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
             (6.283185307179586*Cos(Pi*R)*Sin(P)*Sin(2*P)*
                (4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
                Sin(Pi*Z)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*Power(R,2)*
                Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
             (9.869604401089358*Sin(P)*Sin(2*P)*Sin(Pi*R)*
                (4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
                Sin(Pi*Z)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*R*Power(2. + 
                  1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
             (2*Sin(2*P)*(-330.73361792319804*R*Cos(2*Pi*R) - 
                  78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*Power(R,2)*
                (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
             (2*Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                  52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)\
)/(Sqrt(-mue)*Power(R,3)*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
             (Sin(2*P)*(-826.834044807995*Cos(2*Pi*R) + 
                  2078.060608725385*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
              (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))\
))/R + ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
           (4.1887902047863905*Cos(2*Pi*R) - 
             52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
         (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
        (3.141592653589793*Cos(Pi*Z)*Sin(P)*Sin(2*P)*Sin(Pi*R)*
           (4.1887902047863905*Cos(2*Pi*R) - 
             52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
           Sin(2*Pi*Z))/
         (Sqrt(-mue)*R*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),
            2)))*((taue*(-50. + 1.*R - 0.1*Power(Z,2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
        (beta*(1 + Sqrt(-mue))*Power(Sin(2*P),2)*
           (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
           (4.1887902047863905*Cos(2*Pi*R) - 
             52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
           Power(Sin(2*Pi*Z),2))/
         (30.*mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
           (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
        (taue - (2770.7474783005127*Power(Sin(2*P),2)*
              Power(0.07957747154594767*Cos(2*Pi*R) - 1.*R*Sin(2*Pi*R),2)*
              Power(Sin(2*Pi*t),2)*Power(Sin(2*Pi*Z),2))/
            (Power(R,2)*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),
               2)))/(10.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) \
+ (Sin(3*P)*((0.09424777960769379*mui*taui + 
                Power(R,2)*(0.1884955592153876 + 
                   16.7433894073619*mui*taui))*Cos(3*Pi*R) + 
             0.8882643960980426*mui*R*taui*Sin(3*Pi*R))*Sin(3*Pi*t)*
           Sin(3*Pi*Z))/
         (R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) - 
     (35.06727277224087*beta*(1. + Sqrt(-mue))*Sin(2*P)*Sin(3*P)*Sin(2*Pi*t)*
        Sin(3*Pi*t)*(R*Cos(2*Pi*R)*Cos(3*Pi*Z)*
           (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
             R*(0.011257909293593089 + 1.*mui*taui)*Sin(3*Pi*R))*Sin(2*Pi*Z) \
+ Sin(2*Pi*R)*(R*Sin(3*Pi*R)*((0.0017917519129555279 + 
                   0.15915494309189535*mui*taui)*Cos(3*Pi*Z)*Sin(2*Pi*Z) - 
                0.053051647697298476*mui*taui*Cos(2*Pi*Z)*Sin(3*Pi*Z)) + 
             Cos(3*Pi*R)*(-0.008443431970194816*mui*taui*Cos(3*Pi*Z)*
                 Sin(2*Pi*Z) + 
                (-0.005628954646796544*mui*taui + 
                   Power(R,2)*(-0.011257909293593089 - 1.*mui*taui))*
                 Cos(2*Pi*Z)*Sin(3*Pi*Z)))))/
      (Sqrt(-mue)*mue*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))))
; }};
struct SWi{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return -(Sin(2*P)*(-2.5464790894703255*Cos(2*Pi*R) + 
        31.999999999999996*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
      (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*t)*Sin(P)*Sin(Pi*Z) + 
        Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*t)*Sin(P)*
         Sin(Pi*R)*Sin(Pi*Z))*Sin(2*Pi*Z))/
   (3.*Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
        Sin(Pi*Z) + R*(0.4052847345693511 + 
          (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
           Sin(Pi*Z)),2)) + (2*Pi*Cos(2*Pi*t)*Sin(2*P)*
     ((beta*(1 + Sqrt(-mue))*Sin(2*Pi*R))/(Sqrt(-mue)*mui) + 
       (-2.5464790894703255*Cos(2*Pi*R) + 
          31.999999999999996*R*Sin(2*Pi*R))/
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
              Sin(Pi*t)*Sin(Pi*Z))))*Sin(2*Pi*Z))/3. + 
  FELTORPERP*(0. + (1.623484850566707*beta*(1. + Sqrt(-mue))*taui*Sin(P)*
        Sin(2*P)*Sin(Pi*t)*Sin(2*Pi*t)*Sin(Pi*Z)*
        (0.3183098861837907*mui*R*taui*Power(Cos(Pi*R),3)*
           Power(Cos(Pi*Z),2) + 
          R*(-0.06450306886639899 - 0.6366197723675813*mui*taui)*
           Cos(Pi*R)*Power(Cos(Pi*Z),2)*Power(Sin(Pi*R),2) + 
          Power(Cos(Pi*R),2)*Sin(Pi*R)*
           (0.20264236728467555*mui*taui*Power(Cos(Pi*Z),2) + 
             (-0.10132118364233778*mui*taui + 
                Power(R,2)*(-0.20264236728467555 - 2.*mui*taui))*
              Power(Sin(Pi*Z),2)) + 
          R*Sin(Pi*R)*(R*(0.20264236728467555 + 2.*mui*taui)*
              Power(Cos(Pi*Z),2)*Power(Sin(Pi*R),2) - 
             0.15915494309189535*mui*taui*Sin(2*Pi*R)*Power(Sin(Pi*Z),2))))/
      (Sqrt(-mue)*mui*R*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
           Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
     (0.1*taui*Z*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
          10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
      (Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5)*
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z)))) + ((-33.0733617923198*taui*Cos(2*Pi*Z)*
           Sin(2*P)*(-0.07957747154594767*Cos(2*Pi*R) + 
             1.*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
         Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)) + 
        (5.263789013914324*taui*Z*Sin(2*P)*
           (-0.07957747154594767*Cos(2*Pi*R) + 1.*R*Sin(2*Pi*R))*
           Sin(2*Pi*t)*Sin(2*Pi*Z))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5))/
      (R*(1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
           Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
     (eta*Power(1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z),2)*
        (-((Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
                 52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)\
)/(Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) + 
          (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)))))/
      (mui*(1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
           Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
     (nuperp*(-((Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                 10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
               (0.4052847345693511 + 
                 Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                  Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                 0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                 (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                  Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
             Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                Sin(Pi*t)*Sin(Pi*Z) + 
               R*(0.4052847345693511 + 
                  (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                   Sin(Pi*t)*Sin(Pi*Z)),2)) + 
          (Sin(2*P)*(67.02064327658225*R*Cos(2*Pi*R) + 16.*Sin(2*Pi*R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z))) + 
          R*((2*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                  10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                Power(0.4052847345693511 + 
                  Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                   Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                  0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                  (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                   Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)*Sin(2*Pi*Z))/
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),3) - 
             (2*Sin(2*P)*(67.02064327658225*R*Cos(2*Pi*R) + 
                  16.*Sin(2*Pi*R))*Sin(2*Pi*t)*
                (0.4052847345693511 + 
                  Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                   Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                  0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                  (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                   Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),2) - 
             (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                  10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                (1.5707963267948966*mui*taui*Cos(Pi*R)*Sin(P)*
                   Sin(Pi*t)*Sin(Pi*Z) + 
                  2*Pi*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                   Sin(P)*Sin(Pi*t)*Sin(Pi*Z) - 
                  Power(Pi,2)*R*(0.10132118364233778 + 1.*mui*taui)*
                   Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),2) + 
             (Sin(2*P)*(167.5516081914556*Cos(2*Pi*R) - 
                  421.1031211131459*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                Sin(2*Pi*Z))/
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)))) + 
          R*((-4*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                (-0.8488263631567752*Cos(2*Pi*R) + 
                  10.666666666666666*R*Sin(2*Pi*R))*
                (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
                  Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*
                   Sin(P)*Sin(Pi*R)*Sin(Pi*t))*Sin(2*Pi*t))/
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),2) + 
             (2*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                  10.666666666666666*R*Sin(2*Pi*R))*
                Power(-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                   Sin(Pi*t) + 
                  Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*
                   Sin(P)*Sin(Pi*R)*Sin(Pi*t),2)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),3) - 
             (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                  10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                (1.5707963267948966*mui*taui*Cos(Pi*R)*Sin(P)*
                   Sin(Pi*t)*Sin(Pi*Z) - 
                  Power(Pi,2)*R*(0.10132118364233778 + 1.*mui*taui)*
                   Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
              Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                 Sin(Pi*t)*Sin(Pi*Z) + 
                R*(0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z)),2) - 
             (4*Power(Pi,2)*Sin(2*P)*
                (-0.8488263631567752*Cos(2*Pi*R) + 
                  10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                Sin(2*Pi*Z))/
              (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                 Sin(Pi*Z) + R*
                 (0.4052847345693511 + 
                   (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                    Sin(Pi*t)*Sin(Pi*Z))))))/R - 
     (Sin(2*P)*Sin(3*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
          10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(3*Pi*t)*
        Sin(2*Pi*Z)*(120*Pi*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
          (mui*Cos(3*Pi*Z)*Sin(3*P)*
             (528430.5031318273*
                Power((0.005628954646796544*mui*taui + 
                     Power(R,2)*(0.011257909293593089 + 1.*mui*taui)\
)*Cos(3*Pi*R) + 0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
               54*Power(Pi,3)*Power(R,2)*
                Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                  R*(-0.20000000000000004 - 
                     17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
             Sin(3*Pi*t)*Sin(3*Pi*Z))/
           (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
          (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
             (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                  R*(-0.20000000000000004 - 
                     17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
               28034.10888465587*
                Power((0.005628954646796544*mui*taui + 
                     Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                   Cos(3*Pi*R) + 
                  0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                Power(Sin(3*Pi*Z),2)))/
           (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),
              2))))/
      (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z)))) + (-((Sin(2*P)*
             (-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
             (0.4052847345693511 + 
               Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
               0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
              Sin(Pi*t)*Sin(Pi*Z) + 
             R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),2)) + 
        (Sin(2*P)*(67.02064327658225*R*Cos(2*Pi*R) + 16.*Sin(2*Pi*R))*
           Sin(2*Pi*t)*Sin(2*Pi*Z))/
         (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
            Sin(Pi*Z) + R*(0.4052847345693511 + 
              (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
               Sin(Pi*t)*Sin(Pi*Z))))*
      ((-0.1*R*taui*Z)/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) + 
        (beta*(1 + Sqrt(-mue))*Pi*R*Cos(2*Pi*Z)*Power(Sin(2*P),2)*
           Sin(2*Pi*R)*(-0.8488263631567752*Cos(2*Pi*R) + 
             10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
           Sin(2*Pi*Z))/
         (15.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 
             1.*Power(Z,2))*(-0.15915494309189535*mui*taui*Cos(Pi*R)*
              Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
             R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)))) - 
        (R*Sin(3*P)*Sin(3*Pi*t)*
           (120*Pi*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
             (mui*Cos(3*Pi*Z)*Sin(3*P)*
                (528430.5031318273*
                   Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                      Cos(3*Pi*R) + 
                     0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
                  54*Power(Pi,3)*Power(R,2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                Sin(3*Pi*t)*Sin(3*Pi*Z))/
              (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) \
+ (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                  28034.10888465587*
                   Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                      Cos(3*Pi*R) + 
                     0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                   Power(Sin(3*Pi*Z),2)))/
              (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                  1.*Power(Z,2),2))))/
         (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))) + 
     (beta*(1 + Sqrt(-mue))*Sin(2*P)*Sin(3*P)*Sin(2*Pi*t)*Sin(3*Pi*t)*
        (-((2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*Sin(2*Pi*Z)*
             (120*Pi*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
               (mui*Cos(3*Pi*Z)*Sin(3*P)*
                  (528430.5031318273*
                     Power((0.005628954646796544*mui*taui + 
                       Power(R,2)*
                       (0.011257909293593089 + 1.*mui*taui))*
                       Cos(3*Pi*R) + 
                       0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
                    54*Power(Pi,3)*Power(R,2)*
                     Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                       R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
                  Sin(3*Pi*t)*Sin(3*Pi*Z))/
                (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 
                    1.*Power(Z,2))) + 
               (2.*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
                  (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                     Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                       R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                    28034.10888465587*
                     Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                        Cos(3*Pi*R) + 
                       0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                     Power(Sin(3*Pi*Z),2)))/
                (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                    1.*Power(Z,2),2)))) + 
          2*Pi*R*Cos(2*Pi*Z)*Sin(2*Pi*R)*
           (120*Pi*Cos(3*Pi*R)*Sin(3*Pi*Z) + 
             (mui*(-20. + 2.*R)*Sin(3*P)*Sin(3*Pi*t)*
                (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                  28034.10888465587*
                   Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                      Cos(3*Pi*R) + 
                     0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                   Power(Sin(3*Pi*Z),2)))/
              (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                  1.*Power(Z,2),2)) + 
             (2*mui*Sin(3*P)*Sin(3*Pi*t)*
                (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                  28034.10888465587*
                   Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                      Cos(3*Pi*R) + 
                     0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                   Power(Sin(3*Pi*Z),2)))/
              (Power(R,3)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) \
- (mui*Sin(3*P)*Sin(3*Pi*t)*(18*Power(Pi,2)*R*Power(Cos(3*Pi*Z),2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                  528430.5031318273*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                   (R*(0.011257909293593089 + 1.*mui*taui)*
                      Cos(3*Pi*R) + 
                     (0.0011945012753036852 + 
                        0.15915494309189537*mui*taui)*Sin(3*Pi*R))*
                   (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
                     R*(0.011257909293593089 + 1.*mui*taui)*Sin(3*Pi*R)) \
- 168204.65330793522*((0.005628954646796544*mui*taui + 
                        Power(R,2)*(0.011257909293593089 + 1.*mui*taui)\
)*Cos(3*Pi*R) + 0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                   (R*(-0.007505272862395392 - 
                        0.8333333333333334*mui*taui)*Cos(3*Pi*R) + 
                     (-6.938893903907228e-18*mui*taui + 
                        Power(R,2)*
                         (0.03536776513153231 + 
                         3.141592653589793*mui*taui))*Sin(3*Pi*R))*
                   Power(Sin(3*Pi*Z),2)))/
              (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))\
)/(6000.*Sqrt(-mue)*mui*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) \
+ ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
           (-0.8488263631567752*Cos(2*Pi*R) + 
             10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
         (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
            Sin(Pi*Z) + R*(0.4052847345693511 + 
              (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
               Sin(Pi*t)*Sin(Pi*Z))) - 
        (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
             10.666666666666666*R*Sin(2*Pi*R))*
           (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
             Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*Sin(P)*
              Sin(Pi*R)*Sin(Pi*t))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
         Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
            Sin(Pi*Z) + R*(0.4052847345693511 + 
              (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
               Sin(Pi*t)*Sin(Pi*Z)),2))*
      ((taui*(-50. + 1.*R - 0.1*Power(Z,2)))/
         Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),1.5) - 
        (beta*(1 + Sqrt(-mue))*Power(Sin(2*P),2)*
           (2*Pi*R*Cos(2*Pi*R) + Sin(2*Pi*R))*
           (-0.8488263631567752*Cos(2*Pi*R) + 
             10.666666666666666*R*Sin(2*Pi*R))*Power(Sin(2*Pi*t),2)*
           Power(Sin(2*Pi*Z),2))/
         (30.*Sqrt(-mue)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)))) - 
        (taui + (mui*Power(Sin(2*P),2)*
              Power(0.8488263631567752*Cos(2*Pi*R) - 
                10.666666666666666*R*Sin(2*Pi*R),2)*Power(Sin(2*Pi*t),2)*
              Power(Sin(2*Pi*Z),2))/
            Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z) + 
              R*(0.4052847345693511 + 
                 (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                  Sin(Pi*t)*Sin(Pi*Z)),2))/
         (10.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) + 
        (R*Sin(3*P)*Sin(3*Pi*t)*
           (120*Pi*Cos(3*Pi*R)*Sin(3*Pi*Z) + 
             (mui*(-20. + 2.*R)*Sin(3*P)*Sin(3*Pi*t)*
                (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                  28034.10888465587*
                   Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                      Cos(3*Pi*R) + 
                     0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                   Power(Sin(3*Pi*Z),2)))/
              (Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                  1.*Power(Z,2),2)) + 
             (2*mui*Sin(3*P)*Sin(3*Pi*t)*
                (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                  28034.10888465587*
                   Power((0.005628954646796544*mui*taui + 
                        Power(R,2)*
                        (0.011257909293593089 + 1.*mui*taui))*
                      Cos(3*Pi*R) + 
                     0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                   Power(Sin(3*Pi*Z),2)))/
              (Power(R,3)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))) - 
             (mui*Sin(3*P)*Sin(3*Pi*t)*
                (18*Power(Pi,2)*R*Power(Cos(3*Pi*Z),2)*
                   Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                     R*(-0.20000000000000004 - 
                        17.765287921960844*mui*taui)*Sin(3*Pi*R),2) + 
                  528430.5031318273*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                   (R*(0.011257909293593089 + 1.*mui*taui)*Cos(3*Pi*R) + 
                     (0.0011945012753036852 + 
                        0.15915494309189537*mui*taui)*Sin(3*Pi*R))*
                   (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
                     R*(0.011257909293593089 + 1.*mui*taui)*Sin(3*Pi*R)) - 
                  168204.65330793522*
                   ((0.005628954646796544*mui*taui + 
                        Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                      Cos(3*Pi*R) + 
                     0.05305164769729846*mui*R*taui*Sin(3*Pi*R))*
                   (R*(-0.007505272862395392 - 
                        0.8333333333333334*mui*taui)*Cos(3*Pi*R) + 
                     (-6.938893903907228e-18*mui*taui + 
                        Power(R,2)*
                         (0.03536776513153231 + 
                         3.141592653589793*mui*taui))*Sin(3*Pi*R))*
                   Power(Sin(3*Pi*Z),2)))/
              (Power(R,2)*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2)))))/
         (2000.*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))))) + 
  FELTORPARALLEL*((taui*Sin(Pi*t)*
        (R*Sin(Pi*R)*(R*(7.853981633974482 + 
                77.51569170074954*mui*taui + 
                R*(-0.7853981633974483 - 7.751569170074954*mui*taui))*
              Cos(Pi*Z)*Sin(P) + 
             ((5. + 49.34802200544679*mui*taui)*Cos(P) + 
                1.2337005501361697*mui*taui*Z*Sin(P))*Sin(Pi*Z)) + 
          Cos(Pi*R)*(mui*R*(-12.337005501361698 + 1.2337005501361697*R)*
              taui*Cos(Pi*Z)*Sin(P) + 
             (-7.853981633974484*mui*taui*Cos(P) + 
                (0.39269908169872414*mui*taui + 
                   Power(R,2)*
                    (0.7853981633974483 + 7.751569170074954*mui*taui))*Z*
                 Sin(P))*Sin(Pi*Z))))/
      (mui*Power(R,2)*Sqrt(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
           Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
     (1.*(-10 + R)*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
          10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
             (-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z))) - 
          (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*
             (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
               Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*Sin(P)*
                Sin(Pi*R)*Sin(Pi*t))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),2)))/
      (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z)))) + (1.*Z*Sin(2*P)*
        (-0.8488263631567752*Cos(2*Pi*R) + 
          10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        (-((Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                 10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
               (0.4052847345693511 + 
                 Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                  Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                 0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                 (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                  Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
             Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                Sin(Pi*t)*Sin(Pi*Z) + 
               R*(0.4052847345693511 + 
                  (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                   Sin(Pi*t)*Sin(Pi*Z)),2)) + 
          (Sin(2*P)*(67.02064327658225*R*Cos(2*Pi*R) + 16.*Sin(2*Pi*R))*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)))))/
      (Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z)))) + (20*Sin(2*P)*
        (-0.8488263631567752*Cos(2*Pi*R) + 
          10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z)*
        (-((Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                 10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
               (-0.15915494309189535*mui*taui*Cos(P)*Cos(Pi*R)*
                  Sin(Pi*t)*Sin(Pi*Z) + 
                 R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*Sin(Pi*R)*
                  Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
             Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                Sin(Pi*t)*Sin(Pi*Z) + 
               R*(0.4052847345693511 + 
                  (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                   Sin(Pi*t)*Sin(Pi*Z)),2)) + 
          (2*Cos(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)))))/
      (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))*
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z)))) - (nuparalleli*
        (0. + (1.*R*Z*((-1.*(-10 + R)*
                  ((-2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                        Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
                    (2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                       Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (Power(Pi,2)*R*
                       (0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
                        1.5707963267948966*mui*taui*Cos(Pi*Z)*
                       Sin(P)*Sin(Pi*R)*Sin(Pi*t) + 
                        Pi*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (1.*Power(-10 + R,2)*
                  ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
               (1.*((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (20*((2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                       Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(P)*Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Cos(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                       Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (2*Cos(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) - 
               (1.*(-10 + R)*Z*
                  (-((Sin(2*P)*
                        (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) + 
               (1.*Z*((2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Power(0.4052847345693511 + 
                       Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                       (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                       Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (2*Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (1.5707963267948966*mui*taui*Cos(Pi*R)*
                       Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        2*Pi*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) - 
                        Power(Pi,2)*R*
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (Sin(2*P)*
                       (167.5516081914556*Cos(2*Pi*R) - 
                        421.1031211131459*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
               (20.*(-10 + R)*
                  (-((Sin(2*P)*
                        (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (-0.15915494309189535*mui*taui*Cos(P)*
                       Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) \
- (20*(-((Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (-0.15915494309189535*mui*taui*Cos(P)*
                        Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (Power(R,2)*Sqrt(400 + 1.*Power(-10 + R,2) + 
                    1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (1.*Z*((-1.*(-10 + R)*
                  ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (1.*Z*(-((Sin(2*P)*
                        (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (20*(-((Sin(2*P)*
                        (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (-0.15915494309189535*mui*taui*Cos(P)*
                        Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
          (20*((-1.*(-10 + R)*
                  ((-2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                        Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
                    (2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                       Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                       Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(P)*Cos(Pi*R)*Cos(Pi*Z)*
                       Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(P)*Cos(Pi*Z)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (1.*Z*((2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                       Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(P)*Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Cos(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                       Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                       Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                       Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (2*Cos(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (20*((2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Power(-0.15915494309189535*mui*taui*Cos(P)*
                       Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (4*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                        Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) - 
                        R*(0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (4*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
          (1.*(-10 + R)*R*((1.*Z*
                  ((-2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                        Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
                    (2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*
                       (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                        Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (Power(Pi,2)*R*
                        (0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
                        1.5707963267948966*mui*taui*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t) + 
                        Pi*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) + 
               (20*((-2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                        Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (4*Pi*Cos(2*P)*Cos(2*Pi*Z)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
                    (2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*
                       (-0.15915494309189535*mui*taui*Cos(P)*
                        Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(P)*Cos(Pi*R)*Cos(Pi*Z)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(P)*Cos(Pi*Z)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                (R*Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))) + 
               (1.*(-10 + R)*Z*
                  ((2*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) - 
               (1.*Power(Z,2)*
                  (-((Sin(2*P)*
                        (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                        Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5) + 
               (1.*(-((Sin(2*P)*
                        (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (0.4052847345693511 + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
                        Sin(Pi*Z) + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (Sin(2*P)*
                       (67.02064327658225*R*Cos(2*Pi*R) + 
                        16.*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2)) - 
               (20.*Z*(-((Sin(2*P)*
                        (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                        (-0.15915494309189535*mui*taui*Cos(P)*
                        Cos(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.10132118364233778 + 1.*mui*taui)*Cos(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                       Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                        R*(0.4052847345693511 + 
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2)) + 
                    (2*Cos(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                (R*Power(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2),1.5)) - 
               (1.*(-10 + R)*((-4*Pi*Cos(2*Pi*Z)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*
                        Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t))*
                       Sin(2*Pi*t))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) + 
                    (2*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*
                       Power(-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*
                        Sin(P)*Sin(Pi*t) + 
                        Pi*R*(0.10132118364233778 + 1.*mui*taui)*
                        Cos(Pi*Z)*Sin(P)*Sin(Pi*R)*Sin(Pi*t),2)*
                       Sin(2*Pi*t)*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),3) - 
                    (Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                        10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       (1.5707963267948966*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) - 
                        Power(Pi,2)*R*
                        (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                        Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
                     Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*
                        Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)),2) - 
                    (4*Power(Pi,2)*Sin(2*P)*
                       (-0.8488263631567752*Cos(2*Pi*R) + 
                         10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
                       Sin(2*Pi*Z))/
                     (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
                        Sin(Pi*t)*Sin(Pi*Z) + 
                       R*(0.4052847345693511 + 
                         (0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                         Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))))/
                Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
           Sqrt(400 + 1.*Power(-10 + R,2) + 1.*Power(Z,2))))/
      (R*(1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z))/R + (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*
           Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
     (Sin(3*Pi*t)*(-0.3*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
           Cos(3*P)*(-40.*Power(R,2)*
              (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Sin(3*Pi*R)*
              Sin(3*Pi*Z) + 2*mui*Sin(3*P)*Sin(3*Pi*t)*
              (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                 Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                   R*(-0.20000000000000004 - 17.765287921960844*mui*taui)*
                    Sin(3*Pi*R),2) + 
                28034.10888465587*
                 Power((0.005628954646796544*mui*taui + 
                      Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                    Cos(3*Pi*R) + 
                   0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                 Power(Sin(3*Pi*Z),2))) + 
          0.6*(10 - R)*R*Sin(3*P)*
           (Pi*Power(R,2)*Power(500. - 20.*R + 1.*Power(R,2) + 
                1.*Power(Z,2),2)*Cos(3*Pi*Z)*Sin(3*Pi*R) - 
             0.008333333333333333*mui*
              (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(3*Pi*Z)*
              Sin(3*P)*(528430.5031318273*
                 Power((0.005628954646796544*mui*taui + 
                      Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                    Cos(3*Pi*R) + 
                   0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2) - 
                54*Power(Pi,3)*Power(R,2)*
                 Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                   R*(-0.20000000000000004 - 
                      17.765287921960844*mui*taui)*Sin(3*Pi*R),2))*
              Sin(3*Pi*t)*Sin(3*Pi*Z) + 
             0.016666666666666666*mui*Z*Sin(3*P)*Sin(3*Pi*t)*
              (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                 Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                   R*(-0.20000000000000004 - 17.765287921960844*mui*taui)*
                    Sin(3*Pi*R),2) + 
                28034.10888465587*
                 Power((0.005628954646796544*mui*taui + 
                      Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                    Cos(3*Pi*R) + 
                   0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                 Power(Sin(3*Pi*Z),2))) + 
          0.005*Z*Sin(3*P)*(376.99111843077515*Power(R,3)*
              Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2)*
              Cos(3*Pi*R)*Sin(3*Pi*Z) - 
             mui*(20. - 2.*R)*R*Sin(3*P)*Sin(3*Pi*t)*
              (9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                 Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                   R*(-0.20000000000000004 - 17.765287921960844*mui*taui)*
                    Sin(3*Pi*R),2) + 
                28034.10888465587*
                 Power((0.005628954646796544*mui*taui + 
                      Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                    Cos(3*Pi*R) + 
                   0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                 Power(Sin(3*Pi*Z),2)) + 
             2*mui*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Sin(3*P)*
              Sin(3*Pi*t)*(9*Power(Pi,2)*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                 Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                   R*(-0.20000000000000004 - 17.765287921960844*mui*taui)*
                    Sin(3*Pi*R),2) + 
                28034.10888465587*
                 Power((0.005628954646796544*mui*taui + 
                      Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                    Cos(3*Pi*R) + 
                   0.05305164769729846*mui*R*taui*Sin(3*Pi*R),2)*
                 Power(Sin(3*Pi*Z),2)) - 
             mui*R*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Sin(3*P)*
              Sin(3*Pi*t)*(18*Power(Pi,2)*R*Power(Cos(3*Pi*Z),2)*
                 Power(0.942477796076938*mui*taui*Cos(3*Pi*R) + 
                   R*(-0.20000000000000004 - 17.765287921960844*mui*taui)*
                    Sin(3*Pi*R),2) + 
                528430.5031318273*Power(R,2)*Power(Cos(3*Pi*Z),2)*
                 (R*(0.011257909293593089 + 1.*mui*taui)*Cos(3*Pi*R) + 
                   (0.0011945012753036852 + 0.15915494309189537*mui*taui)*
                    Sin(3*Pi*R))*
                 (-0.053051647697298455*mui*taui*Cos(3*Pi*R) + 
                   R*(0.011257909293593089 + 1.*mui*taui)*Sin(3*Pi*R)) - 
                168204.65330793522*
                 ((0.005628954646796544*mui*taui + 
                      Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*
                    Cos(3*Pi*R) + 0.05305164769729846*mui*R*taui*Sin(3*Pi*R)\
)*(R*(-0.007505272862395392 - 0.8333333333333334*mui*taui)*Cos(3*Pi*R) + 
                   (-6.938893903907228e-18*mui*taui + 
                      Power(R,2)*
                       (0.03536776513153231 + 3.141592653589793*mui*taui))*
                    Sin(3*Pi*R))*Power(Sin(3*Pi*Z),2)))))/
      (mui*Power(R,3)*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2.5))\
)
; }};
struct SPhie{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 0.25*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
  (Sin(3*P)*Sin(3*Pi*t)*(0.2191704548265055*R*
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*Cos(Pi*Z)*
        Cos(3*Pi*Z)*Sin(P)*(-0.5*mui*taui*Cos(Pi*R) + 
          R*(0.3183098861837907 + 3.141592653589793*mui*taui)*Sin(Pi*R))*
        (1.*mui*taui*Cos(3*Pi*R) + 
          R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*Sin(3*Pi*R)\
)*Sin(Pi*t) - 0.438340909653011*R*Z*Cos(3*Pi*Z)*
        (1.*mui*taui*Cos(3*Pi*R) + 
          R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*Sin(3*Pi*R)\
)*(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
          R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z))) - 4.131265744601299*
        (500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        ((0.005628954646796544*mui*taui + 
             Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*Cos(3*Pi*R) \
+ 0.053051647697298476*mui*R*taui*Sin(3*Pi*R))*
        (0.4052847345693511 + R*
           (0.3183098861837907 + 3.141592653589793*mui*taui)*Cos(Pi*R)*
           Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
          (0.10132118364233778 + 1.5*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
           Sin(Pi*Z))*Sin(3*Pi*Z) + 
       4.131265744601299*(-20. + 2.*R)*
        ((0.005628954646796544*mui*taui + 
             Power(R,2)*(0.011257909293593089 + 1.*mui*taui))*Cos(3*Pi*R) \
+ 0.053051647697298476*mui*R*taui*Sin(3*Pi*R))*
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
              Sin(Pi*t)*Sin(Pi*Z)))*Sin(3*Pi*Z) - 
       2.06563287230065*R*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (1.*mui*taui*Cos(3*Pi*R) + 
          R*(-0.2122065907891938 - 18.849555921538755*mui*taui)*Sin(3*Pi*R)\
)*(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
          R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
              Sin(Pi*t)*Sin(Pi*Z)))*Sin(3*Pi*Z) - 
       4.131265744601299*(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2))*
        (R*(0.022515818587186178 + 2.5*mui*taui)*Cos(3*Pi*R) + 
          (3.122502256758253e-17*mui*taui + 
             Power(R,2)*(-0.10610329539459692 - 9.42477796076938*mui*taui))*
           Sin(3*Pi*R))*(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
           Sin(Pi*t)*Sin(Pi*Z) + 
          R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z)))*Sin(3*Pi*Z)))/
   (R*Power(500. - 20.*R + 1.*Power(R,2) + 1.*Power(Z,2),2))
; }};
struct Snehat{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli,alpha;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1 + 0.5*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*Sin(Pi*Z) + 
  (alpha*nuperp*(0. + 1.5707963267948966*Cos(Pi*(-10 + R))*Sin(P)*Sin(Pi*t)*
        Sin(Pi*Z) - 9.869604401089358*R*Sin(P)*Sin(Pi*(-10 + R))*Sin(Pi*t)*
        Sin(Pi*Z)))/R
; }};
struct SNihat{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli,alpha;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return 1. - (0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/R + 
  (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
  (alpha*nuperp*((0.39269908169872414*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
          Sin(Pi*Z))/Power(R,2) + 
       Pi*(0.25 + 2.4674011002723395*mui*taui)*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
        Sin(Pi*Z) + (1.2337005501361697*mui*taui*Sin(P)*Sin(Pi*R)*
          Sin(Pi*t)*Sin(Pi*Z))/R + 
       R*((3.875784585037477*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z))/R - Power(Pi,2)*
           (0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
           Sin(Pi*Z)) + R*((-0.7853981633974483*mui*taui*Cos(Pi*R)*Sin(P)*
             Sin(Pi*t)*Sin(Pi*Z))/Power(R,3) + 
          (3.875784585037477*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z))/
           R - (2.4674011002723395*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
             Sin(Pi*Z))/Power(R,2) - 
          Power(Pi,2)*(0.25 + 2.4674011002723395*mui*taui)*Sin(P)*Sin(Pi*R)*
           Sin(Pi*t)*Sin(Pi*Z))))/R
; }};
struct SWehat{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli,alpha;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*t)*((beta*(1 + Sqrt(-mue))*Sin(2*Pi*R))/mue + 
       (12.566370614359172*Cos(2*Pi*R) - 157.91367041742973*R*Sin(2*Pi*R))/
        (R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))*Sin(2*Pi*Z))/
   (3.*Sqrt(-mue)) + (alpha*nuperp*
     ((-3.141592653589793*Cos(Pi*R)*Sin(P)*Sin(2*P)*
          (4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
          Sin(Pi*Z)*Sin(2*Pi*Z))/
        (Sqrt(-mue)*R*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) \
+ (Sin(2*P)*(-330.73361792319804*R*Cos(2*Pi*R) - 
            78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
        (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) - 
       (Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
            52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
        (Sqrt(-mue)*Power(R,2)*
          (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
       R*((-39.47841760435743*Cos(Pi*Z)*Cos(2*Pi*Z)*Sin(P)*Sin(2*P)*
             Sin(Pi*R)*(4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t))/
           (Sqrt(-mue)*R*Power(2. + 
               1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
          (19.739208802178716*Power(Cos(Pi*Z),2)*Power(Sin(P),2)*Sin(2*P)*
             Power(Sin(Pi*R),2)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(Pi*t),2)*
             Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 
               1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),3)) + 
          (9.869604401089358*Sin(P)*Sin(2*P)*Sin(Pi*R)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(Pi*Z)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 
               1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) - 
          (4*Power(Pi,2)*Sin(2*P)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z)))) + 
       R*((19.739208802178716*Power(Cos(Pi*R),2)*Power(Sin(P),2)*Sin(2*P)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Power(Sin(Pi*t),2)*
             Sin(2*Pi*t)*Power(Sin(Pi*Z),2)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),
              3)) - (6.283185307179586*Cos(Pi*R)*Sin(P)*Sin(2*P)*
             (-330.73361792319804*R*Cos(2*Pi*R) - 
               78.95683520871486*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(Pi*Z)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),
              2)) + (6.283185307179586*Cos(Pi*R)*Sin(P)*Sin(2*P)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(Pi*Z)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*Power(R,2)*
             Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),2)) + 
          (9.869604401089358*Sin(P)*Sin(2*P)*Sin(Pi*R)*
             (4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(Pi*t)*Sin(2*Pi*t)*
             Sin(Pi*Z)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*Power(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z),
              2)) - (2*Sin(2*P)*
             (-330.73361792319804*R*Cos(2*Pi*R) - 
               78.95683520871486*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*Power(R,2)*
             (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
          (2*Sin(2*P)*(4.1887902047863905*Cos(2*Pi*R) - 
               52.63789013914324*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*Power(R,3)*
             (2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))) + 
          (Sin(2*P)*(-826.834044807995*Cos(2*Pi*R) + 
               2078.060608725385*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (Sqrt(-mue)*R*(2. + 1.*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))))))/R
; }};
struct SWihat{
    double mue,mui,taue,taui,eta,beta,nuperp,nuparallele,nuparalleli,alpha;
    DG_DEVICE double operator()(double R, double Z, double P, double t)const{
    return (Sin(2*P)*Sin(2*Pi*t)*((beta*(1 + Sqrt(-mue))*Sin(2*Pi*R))/
        (Sqrt(-mue)*mui) + (-2.5464790894703255*Cos(2*Pi*R) + 
          31.999999999999996*R*Sin(2*Pi*R))/
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
              Sin(Pi*t)*Sin(Pi*Z))))*Sin(2*Pi*Z))/3. + 
  (alpha*nuperp*(-((Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
              10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
            (0.4052847345693511 + 
              Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*Sin(P)*
               Sin(Pi*t)*Sin(Pi*Z) + 
              0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
              (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
               Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
          Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
             Sin(Pi*Z) + R*(0.4052847345693511 + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z)),2)) + 
       (Sin(2*P)*(67.02064327658225*R*Cos(2*Pi*R) + 16.*Sin(2*Pi*R))*
          Sin(2*Pi*t)*Sin(2*Pi*Z))/
        (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
           Sin(Pi*Z) + R*(0.4052847345693511 + 
             (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*
              Sin(Pi*Z))) + R*((2*Sin(2*P)*
             (-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
             Power(0.4052847345693511 + 
               Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
               0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z),2)*Sin(2*Pi*Z))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
              Sin(Pi*t)*Sin(Pi*Z) + 
             R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),3) - 
          (2*Sin(2*P)*(67.02064327658225*R*Cos(2*Pi*R) + 16.*Sin(2*Pi*R))*
             Sin(2*Pi*t)*(0.4052847345693511 + 
               Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*R)*
                Sin(P)*Sin(Pi*t)*Sin(Pi*Z) + 
               0.5*mui*taui*Sin(P)*Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z) + 
               (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
              Sin(Pi*t)*Sin(Pi*Z) + 
             R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),2) - 
          (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
             (1.5707963267948966*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                Sin(Pi*Z) + 2*Pi*(0.10132118364233778 + 1.*mui*taui)*
                Cos(Pi*R)*Sin(P)*Sin(Pi*t)*Sin(Pi*Z) - 
               Power(Pi,2)*R*(0.10132118364233778 + 1.*mui*taui)*Sin(P)*
                Sin(Pi*R)*Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*
              Sin(Pi*t)*Sin(Pi*Z) + 
             R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),2) + 
          (Sin(2*P)*(167.5516081914556*Cos(2*Pi*R) - 
               421.1031211131459*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)))) + 
       R*((-4*Pi*Cos(2*Pi*Z)*Sin(2*P)*
             (-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*
             (-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
               Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*Sin(P)*
                Sin(Pi*R)*Sin(Pi*t))*Sin(2*Pi*t))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),2) + 
          (2*Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*
             Power(-0.5*mui*taui*Cos(Pi*R)*Cos(Pi*Z)*Sin(P)*Sin(Pi*t) + 
               Pi*R*(0.10132118364233778 + 1.*mui*taui)*Cos(Pi*Z)*Sin(P)*
                Sin(Pi*R)*Sin(Pi*t),2)*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),3) - 
          (Sin(2*P)*(-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*
             (1.5707963267948966*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
                Sin(Pi*Z) - Power(Pi,2)*R*
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                Sin(Pi*t)*Sin(Pi*Z))*Sin(2*Pi*Z))/
           Power(-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z)),2) - 
          (4*Power(Pi,2)*Sin(2*P)*
             (-0.8488263631567752*Cos(2*Pi*R) + 
               10.666666666666666*R*Sin(2*Pi*R))*Sin(2*Pi*t)*Sin(2*Pi*Z))/
           (-0.15915494309189535*mui*taui*Cos(Pi*R)*Sin(P)*Sin(Pi*t)*
              Sin(Pi*Z) + R*(0.4052847345693511 + 
                (0.10132118364233778 + 1.*mui*taui)*Sin(P)*Sin(Pi*R)*
                 Sin(Pi*t)*Sin(Pi*Z))))))/R
; }};
}}//namespace feltor namespace manufactured
