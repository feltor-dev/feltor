#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
//6 analytical quantities
namespace solovev
{
/**
 * @brief Constructs and display geometric parameters
 */    
struct GeomParameters
{
    double A,R_0,psipmin,psipmax,a, elongation,triangularity,alpha;
    std::vector<double> c; 
     /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */   
    GeomParameters( const std::vector< double>& v):layout_(0) {
        if( layout_ == 0)
        {
            A=v[1];
            c.resize(13);
            for (unsigned i=0;i<12;i++) c[i]=v[i+2];
            R_0 = v[14];
            psipmin= v[15];
            psipmax= v[16];
            a=R_0*v[17];
            elongation=v[18];
            triangularity=v[19];
            alpha=v[20];
        }
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Geometrical parameters are: \n"
            <<"A             = "<<A<<"\n"
            <<"c1            = "<<c[0]<<"\n"
            <<"c2            = "<<c[1]<<"\n"
            <<"c3            = "<<c[2]<<"\n"
            <<"c4            = "<<c[3]<<"\n"
            <<"c5            = "<<c[4]<<"\n"
            <<"c6            = "<<c[5]<<"\n"
            <<"c7            = "<<c[6]<<"\n"
            <<"c8            = "<<c[7]<<"\n"
            <<"c9            = "<<c[8]<<"\n"
            <<"c10           = "<<c[9]<<"\n"
            <<"c11           = "<<c[10]<<"\n"
            <<"c12           = "<<c[11]<<"\n"
            <<"R0            = "<<R_0<<"\n"
            <<"psipmin       = "<<psipmin<<"\n"
            <<"psipmax       = "<<psipmax<<"\n"
            <<"epsilon_a     = "<<a/R_0<<"\n"
            <<"elongation    = "<<elongation<<"\n"
            <<"triangularity = "<<triangularity<<"\n"
            <<"alpha         = "<<alpha<<"\n";             
    }
    private:
    int layout_;
};
/**
 * @brief Psip
 */    
struct Psip
{
  Psip( double R_0, double A, std::vector<double> c ): R_0_(R_0), A_(A), c_(c) {}
double operator()(double R, double Z)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,Zn6,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; Zn6 = Zn3*Zn3;
     lgRn= log(Rn);
     return   R_0_*( Rn4/8.+ A_ * ( 1./2.* Rn2* lgRn-(Rn4)/8.) 
                    + c_[0] 
            + c_[1]  *Rn2
            + c_[2]  *(Zn2 - Rn2 * lgRn ) 
            + c_[3]  *(Rn4 - 4.* Rn2*Zn2 ) 
            + c_[4]  *(3.* Rn4 * lgRn  -9.*Rn2*Zn2 -12.* Rn2*Zn2 * lgRn + 2.*Zn4)
            + c_[5]  *(Rn4*Rn2-12.* Rn4*Zn2 +8.* Rn2 *Zn4 ) 
            + c_[6]  *(-15.*Rn4*Rn2 * lgRn + 75.* Rn4 *Zn2 + 180.* Rn4*Zn2 * lgRn 
                       -140.*Rn2*Zn4 - 120.* Rn2*Zn4 *lgRn + 8.* Zn6 )
            + c_[7]  *Zn
            + c_[8]  *Rn2*Zn            
                    + c_[9] *(Zn2*Zn - 3.* Rn2*Zn * lgRn)
            + c_[10] *( 3. * Rn4*Zn - 4. * Rn2*Zn3)
            + c_[11] *(-45.* Rn4*Zn + 60.* Rn4*Zn* lgRn - 80.* Rn2*Zn3* lgRn + 8. * Zn5)            
                    );
  }
  double operator()(double R, double Z, double phi)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,Zn6,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; Zn6 = Zn3*Zn3;
     lgRn= log(Rn);
     return   R_0_*( Rn4/8.+ A_ * ( 1./2.* Rn2* lgRn-(Rn4)/8.) 
                    + c_[0] 
            + c_[1]  *Rn2
            + c_[2]  *(Zn2 - Rn2 * lgRn ) 
            + c_[3]  *(Rn4 - 4.* Rn2*Zn2 ) 
            + c_[4]  *(3.* Rn4 * lgRn  -9.*Rn2*Zn2 -12.* Rn2*Zn2 * lgRn + 2.*Zn4)
            + c_[5]  *(Rn4*Rn2-12.* Rn4*Zn2 +8.* Rn2 *Zn4 ) 
            + c_[6]  *(-15.*Rn4*Rn2 * lgRn + 75.* Rn4 *Zn2 + 180.* Rn4*Zn2 * lgRn 
                       -140.*Rn2*Zn4 - 120.* Rn2*Zn4 *lgRn + 8.* Zn6 )
            + c_[7]  *Zn
            + c_[8]  *Rn2*Zn            
                    + c_[9] *(Zn2*Zn - 3.* Rn2*Zn * lgRn)
            + c_[10] *( 3. * Rn4*Zn - 4. * Rn2*Zn3)
            + c_[11] *(-45.* Rn4*Zn + 60.* Rn4*Zn* lgRn - 80.* Rn2*Zn3* lgRn + 8. * Zn5)            
                    );
  }
  void display()
  {
    std::cout << R_0_ <<"  " <<A_ <<"\n";
    std::cout << c_[0] <<"\n";
  }
  private:
  double R_0_, A_;
  std::vector<double> c_;
//   double * c;
};
/**
 * @brief d Psip/ d R
 */ 
struct PsipR
{
  PsipR( double R_0, double A, std::vector<double> c ): R_0_(R_0), A_(A), c_(c) {}
  double operator()(double R, double Z)
  {    
     double Rn,Rn2,Rn3,Rn5,Zn,Zn2,Zn3,Zn4,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;  Rn5 = Rn3*Rn2; 
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
     lgRn= log(Rn);
     return   (Rn3/2. + (Rn/2. - Rn3/2. + Rn*lgRn)* A_ + 
        2.* Rn* c_[1] + (-Rn - 2.* Rn*lgRn)* c_[2] + (4.*Rn3 - 8.* Rn *Zn2)* c_[3] + 
        (3. *Rn3 - 30.* Rn *Zn2 + 12. *Rn3*lgRn -  24.* Rn *Zn2*lgRn)* c_[4]
        + (6 *Rn5 - 48 *Rn3 *Zn2 + 16.* Rn *Zn4)*c_[5]
        + (-15. *Rn5 + 480. *Rn3 *Zn2 - 400.* Rn *Zn4 - 90. *Rn5*lgRn + 
            720. *Rn3 *Zn2*lgRn - 240.* Rn *Zn4*lgRn)* c_[6] + 
        2.* Rn *Zn *c_[8] + (-3. *Rn *Zn - 6.* Rn* Zn*lgRn)* c_[9] + (12. *Rn3* Zn - 8.* Rn *Zn3)* c_[10] + (-120. *Rn3* Zn - 80.* Rn *Zn3 + 240. *Rn3* Zn*lgRn - 
            160.* Rn *Zn3*lgRn) *c_[11]
          );
  }
  double operator()(double R, double Z, double phi)
  {    
     double Rn,Rn2,Rn3,Rn5,Zn,Zn2,Zn3,Zn4,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn; Rn5 = Rn3*Rn2; 
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2;
     lgRn= log(Rn);
     return   (Rn3/2. + (Rn/2. - Rn3/2. + Rn*lgRn)* A_ + 
        2.* Rn* c_[1] + (-Rn - 2.* Rn*lgRn)* c_[2] + (4.*Rn3 - 8.* Rn *Zn2)* c_[3] + 
        (3. *Rn3 - 30.* Rn *Zn2 + 12. *Rn3*lgRn -  24.* Rn *Zn2*lgRn)* c_[4]
        + (6 *Rn5 - 48 *Rn3 *Zn2 + 16.* Rn *Zn4)*c_[5]
        + (-15. *Rn5 + 480. *Rn3 *Zn2 - 400.* Rn *Zn4 - 90. *Rn5*lgRn + 
            720. *Rn3 *Zn2*lgRn - 240.* Rn *Zn4*lgRn)* c_[6] + 
        2.* Rn *Zn *c_[8] + (-3. *Rn *Zn - 6.* Rn* Zn*lgRn)* c_[9] + (12. *Rn3* Zn - 8.* Rn *Zn3)* c_[10] + (-120. *Rn3* Zn - 80.* Rn *Zn3 + 240. *Rn3* Zn*lgRn - 
            160.* Rn *Zn3*lgRn) *c_[11]
          );
  }
  void display()
  {
    std::cout << R_0_ <<"  " <<A_ <<"\n";
    std::cout << c_[0] <<"\n";
  }
  private:
  double R_0_, A_;
  std::vector<double> c_;
};
/**
 * @brief d^2 Psip/ d R^2
 */ 
struct PsipRR
{
  PsipRR( double R_0, double A, std::vector<double> c ): R_0_(R_0), A_(A), c_(c) {}
  double operator()(double R, double Z)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
     lgRn= log(Rn);
     return   1./R_0_*( (3.* Rn2)/2. + (3./2. - (3. *Rn2)/2. +lgRn) *A_ +  2.* c_[1] + (-3. - 2.*lgRn)* c_[2] + (12. *Rn2 - 8. *Zn2) *c_[3] + 
       (21. *Rn2 - 54. *Zn2 + 36. *Rn2*lgRn - 24. *Zn2*lgRn)* c_[4]
       + (30. *Rn4 - 144. *Rn2 *Zn2 + 16.*Zn4)*c_[5] + (-165. *Rn4 + 2160. *Rn2 *Zn2 - 640. *Zn4 - 450. *Rn4*lgRn + 
    2160. *Rn2 *Zn2*lgRn - 240. *Zn4*lgRn)* c_[6] + 
 2.* Zn* c_[8] + (-9. *Zn - 6.* Zn*lgRn) *c_[9] 
 + (36. *Rn2* Zn - 8. *Zn3) *c_[10]
 + (-120. *Rn2* Zn - 240. *Zn3 + 720. *Rn2* Zn*lgRn - 160. *Zn3*lgRn)* c_[11]);
  }
  double operator()(double R, double Z, double phi)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2; 
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
     lgRn= log(Rn);
     return   1./R_0_*( (3.* Rn2)/2. + (3./2. - (3. *Rn2)/2. +lgRn) *A_ +  2.* c_[1] + (-3. - 2.*lgRn)* c_[2] + (12. *Rn2 - 8. *Zn2) *c_[3] + 
       (21. *Rn2 - 54. *Zn2 + 36. *Rn2*lgRn - 24. *Zn2*lgRn)* c_[4]
       + (30. *Rn4 - 144. *Rn2 *Zn2 + 16.*Zn4)*c_[5] + (-165. *Rn4 + 2160. *Rn2 *Zn2 - 640. *Zn4 - 450. *Rn4*lgRn + 
    2160. *Rn2 *Zn2*lgRn - 240. *Zn4*lgRn)* c_[6] + 
 2.* Zn* c_[8] + (-9. *Zn - 6.* Zn*lgRn) *c_[9] 
 + (36. *Rn2* Zn - 8. *Zn3) *c_[10]
 + (-120. *Rn2* Zn - 240. *Zn3 + 720. *Rn2* Zn*lgRn - 160. *Zn3*lgRn)* c_[11]);
  }
  void display()
  {
    std::cout << R_0_ <<"  " <<A_ <<"\n";
    std::cout << c_[0] <<"\n";
  }
  private:
  double R_0_, A_;
  std::vector<double> c_;
};
/**
 * @brief d Psip/ d Z
 */ 
struct PsipZ
{
  PsipZ( double R_0, double A, std::vector<double> c ): R_0_(R_0), A_(A), c_(c) { }
  double operator()(double R, double Z)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2; 
     Zn = Z/R_0_; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; 
     lgRn= log(Rn);

     return   (2.* Zn* c_[2] 
            -  8. *Rn2* Zn* c_[3] +
              ((-18.)*Rn2 *Zn + 8. *Zn3 - 24. *Rn2* Zn*lgRn) *c_[4] 
            + ((-24.) *Rn4* Zn + 32. *Rn2 *Zn3)* c_[5]   
            + (150. *Rn4* Zn - 560. *Rn2 *Zn3 + 48. *Zn5 + 360. *Rn4* Zn*lgRn - 480. *Rn2 *Zn3*lgRn)* c_[6] 
            + c_[7]
            + Rn2 * c_[8]
            + (3. *Zn2 - 3. *Rn2*lgRn)* c_[9]
            + (3. *Rn4 - 12. *Rn2 *Zn2) *c_[10]
            + ((-45.)*Rn4 + 40. *Zn4 + 60. *Rn4*lgRn -  240. *Rn2 *Zn2*lgRn)* c_[11]);
          
  }
  double operator()(double R, double Z, double phi)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2; 
     Zn = Z/R_0_; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; 
     lgRn= log(Rn);

     return   (2.* Zn* c_[2] 
            -  8. *Rn2* Zn* c_[3] +
              ((-18.)*Rn2 *Zn + 8. *Zn3 - 24. *Rn2* Zn*lgRn) *c_[4] 
            + ((-24.) *Rn4* Zn + 32. *Rn2 *Zn3)* c_[5]   
            + (150. *Rn4* Zn - 560. *Rn2 *Zn3 + 48. *Zn5 + 360. *Rn4* Zn*lgRn - 480. *Rn2 *Zn3*lgRn)* c_[6] 
            + c_[7]
            + Rn2 * c_[8]
            + (3. *Zn2 - 3. *Rn2*lgRn)* c_[9]
            + (3. *Rn4 - 12. *Rn2 *Zn2) *c_[10]
            + ((-45.)*Rn4 + 40. *Zn4 + 60. *Rn4*lgRn -  240. *Rn2 *Zn2*lgRn)* c_[11]);
          
  }
  void display()
  {
    std::cout << R_0_ <<"  " <<A_ <<"\n";
    std::cout << c_[0] <<"\n";
  }
  private:
  double R_0_, A_;
  std::vector<double> c_;
};
/**
 * @brief d^2 Psip/ d Z^2
 */ 
struct PsipZZ
{
  PsipZZ( double R_0, double A, std::vector<double> c ): R_0_(R_0), A_(A), c_(c)  {  }
  double operator()(double R, double Z)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2; 
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
     lgRn= log(Rn);
     return   1./R_0_*( 2.* c_[2] - 8. *Rn2* c_[3] + (-18. *Rn2 + 24. *Zn2 - 24. *Rn2*lgRn) *c_[4] + (-24.*Rn4 + 96. *Rn2 *Zn2) *c_[5]
     + (150. *Rn4 - 1680. *Rn2 *Zn2 + 240. *Zn4 + 360. *Rn4*lgRn - 1440. *Rn2 *Zn2*lgRn)* c_[6] + 6.* Zn* c_[9] -  24. *Rn2 *Zn *c_[10] + (160. *Zn3 - 480. *Rn2* Zn*lgRn) *c_[11]);
  }
  double operator()(double R, double Z, double phi)
  {    
     double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2; 
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
     lgRn= log(Rn);
     return   1./R_0_*( 2.* c_[2] - 8. *Rn2* c_[3] + (-18. *Rn2 + 24. *Zn2 - 24. *Rn2*lgRn) *c_[4] + (-24.*Rn4 + 96. *Rn2 *Zn2) *c_[5]
     + (150. *Rn4 - 1680. *Rn2 *Zn2 + 240. *Zn4 + 360. *Rn4*lgRn - 1440. *Rn2 *Zn2*lgRn)* c_[6] + 6.* Zn* c_[9] -  24. *Rn2 *Zn *c_[10] + (160. *Zn3 - 480. *Rn2* Zn*lgRn) *c_[11]);
  }
  void display()
  {
    std::cout << R_0_ <<"  " <<A_ <<"\n";
    std::cout << c_[0] <<"\n";
  }
  private:
  double R_0_, A_;
  std::vector<double> c_;
};
/**
 * @brief d^2 Psip/ d R d Z
 */ 
struct PsipRZ
{
  PsipRZ( double R_0, double A, std::vector<double> c ): R_0_(R_0), A_(A), c_(c) {  }
  double operator()(double R, double Z)
  {    
     double Rn,Rn2,Rn3,Zn,Zn2,Zn3,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn; 
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; 
     lgRn= log(Rn);
     return   1./R_0_*(
              -16.* Rn* Zn* c_[3] + (-60.* Rn* Zn - 48.* Rn* Zn*lgRn)* c_[4] + (-96. *Rn3* Zn + 64.*Rn *Zn3)* c_[5]
            + (960. *Rn3 *Zn - 1600.* Rn *Zn3 + 1440. *Rn3* Zn*lgRn - 960. *Rn *Zn3*lgRn) *c_[6] +  2.* Rn* c_[8] + (-3.* Rn - 6.* Rn*lgRn)* c_[9]
            + (12. *Rn3 - 24.* Rn *Zn2) *c_[10] + (-120. *Rn3 - 240. *Rn *Zn2 + 240. *Rn3*lgRn -   480.* Rn *Zn2*lgRn)* c_[11]
                 );
  }
  double operator()(double R, double Z, double phi)
  {    
     double Rn,Rn2,Rn3,Zn,Zn2,Zn3,lgRn;
     Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn; 
     Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; 
     lgRn= log(Rn);
     return   1./R_0_*(
              -16.* Rn* Zn* c_[3] + (-60.* Rn* Zn - 48.* Rn* Zn*lgRn)* c_[4] + (-96. *Rn3* Zn + 64.*Rn *Zn3)* c_[5]
            + (960. *Rn3 *Zn - 1600.* Rn *Zn3 + 1440. *Rn3* Zn*lgRn - 960. *Rn *Zn3*lgRn) *c_[6] +  2.* Rn* c_[8] + (-3.* Rn - 6.* Rn*lgRn)* c_[9]
            + (12. *Rn3 - 24.* Rn *Zn2) *c_[10] + (-120. *Rn3 - 240. *Rn *Zn2 + 240. *Rn3*lgRn -   480.* Rn *Zn2*lgRn)* c_[11]
                 );
  }
  void display()
  {
    std::cout << R_0_ <<"  " <<A_ <<"\n";
    std::cout << c_[0] <<"\n";
  }
  private:
  double R_0_, A_;
  std::vector<double> c_;
};
/**
 * @brief I
 */ 
struct Ipol
{
  Ipol(  double R_0, double A, Psip psip ):  R_0_(R_0), A_(A), psip_(psip) { }
  double operator()(double R, double Z)
  {    
    return sqrt(2.*A_* psip_(R,Z) /R_0_ + 1.);
  }
  double operator()(double R, double Z, double phi)
  {    
    return sqrt(2.*A_*psip_(R,Z,phi)/R_0_ + 1.);
  }
  void display()
  {
    std::cout<< R_0_ <<"  "  << A_ <<"\n";
  }
  private:
  double A_,R_0_;
  Psip psip_;
};

/**
 * @brief 1/B
 */ 
struct InvB
{
  InvB(  double R_0, Ipol ipol, PsipR psipR, PsipZ psipZ ):  R_0_(R_0), ipol_(ipol), psipR_(psipR), psipZ_(psipZ)  { }
  double operator()(double R, double Z)
  {    
    return R/(R_0_*sqrt(ipol_(R,Z)*ipol_(R,Z) + psipR_(R,Z)*psipR_(R,Z) +psipZ_(R,Z)*psipZ_(R,Z))) ;
  }
  double operator()(double R, double Z, double phi)
  {    
    return R/(R_0_*sqrt(ipol_(R,Z,phi)*ipol_(R,Z,phi) + psipR_(R,Z,phi)*psipR_(R,Z,phi) +psipZ_(R,Z,phi)*psipZ_(R,Z,phi))) ;
  }
  void display() { }
  private:
  double R_0_;
  Ipol ipol_;
  PsipR psipR_;
  PsipZ psipZ_;  
};
/**
 * @brief d B / d R
 */ 
struct BR
{
  BR(double R_0, double A, PsipR psipR, PsipRR psipRR, PsipZ psipZ, PsipRZ psipRZ, InvB invB):  R_0_(R_0), A_(A), psipR_(psipR), psipRR_(psipRR),psipZ_(psipZ) ,psipRZ_(psipRZ), invB_(invB) { }
  double operator()(double R, double Z)
  { 
    double Rn;
    Rn = R/R_0_;
    return -( Rn*Rn/invB_(R,Z)/invB_(R,Z)- Rn *A_*psipR_(R,Z) - R *(psipZ_(R,Z)*psipRZ_(R,Z)+psipR_(R,Z)*psipRR_(R,Z)))/(R*Rn*Rn/invB_(R,Z));
  }
  double operator()(double R, double Z, double phi)
  { 
    double Rn;
    Rn = R/R_0_;
    return -( Rn*Rn/invB_(R,Z,phi)/invB_(R,Z,phi)- Rn *A_*psipR_(R,Z,phi) - R *(psipZ_(R,Z,phi)*psipRZ_(R,Z,phi)+psipR_(R,Z,phi)*psipRR_(R,Z,phi)))/(R*Rn*Rn/invB_(R,Z,phi));
  }
  void display() { }
  private:
  double R_0_;
  double A_;
  PsipR psipR_;
  PsipRR psipRR_;
  PsipZ psipZ_;
  PsipRZ psipRZ_;  
  InvB invB_;
};
/**
 * @brief d B / d Z
 */ 
struct BZ
{
  BZ(double R_0, double A, PsipR psipR,  PsipZ psipZ, PsipZZ psipZZ, PsipRZ psipRZ, InvB invB):  R_0_(R_0), A_(A), psipR_(psipR),psipZ_(psipZ), psipZZ_(psipZZ) ,psipRZ_(psipRZ), invB_(invB) { }
  double operator()(double R, double Z)
  { 
    double Rn;
    Rn = R/R_0_;
    return (A_/R_0_*psipZ_(R,Z) + psipR_(R,Z)*psipRZ_(R,Z)+psipZ_(R,Z)*psipZZ_(R,Z))/(Rn*Rn/invB_(R,Z));
  }
  double operator()(double R, double Z, double phi)
  { 
    double Rn;
    Rn = R/R_0_;
    return (A_/R_0_*psipZ_(R,Z,phi) + psipR_(R,Z,phi)*psipRZ_(R,Z,phi)+psipZ_(R,Z,phi)*psipZZ_(R,Z,phi))/(Rn*Rn/invB_(R,Z,phi));
  }
  void display() { }
  private:
  double R_0_;
  double A_;
  PsipR psipR_;
  PsipZ psipZ_;
  PsipZZ psipZZ_;
  PsipRZ psipRZ_;  
  InvB invB_; 
};
/**
 * @brief K^R
 */ 
struct CurvatureR
{
    CurvatureR( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)),
        psipR_(PsipR(gp.R_0,gp.A,gp.c)),
        psipZ_(PsipZ(gp.R_0,gp.A,gp.c)),
        psipZZ_(PsipZZ(gp.R_0,gp.A,gp.c)),
        psipRZ_(PsipRZ(gp.R_0,gp.A,gp.c)),
        ipol_(Ipol(gp.R_0,gp.A,psip_)),
        invB_(InvB(gp.R_0,ipol_,psipR_,psipZ_)),
        bZ_(BZ(gp.R_0,gp.A,psipR_,psipZ_,psipZZ_,psipRZ_,invB_)) {
    }
//     CurvatureR( InvB invB, BZ bZ):invB_(invB), bZ_(bZ){}
    double operator()( double R, double Z)
    {
        return -invB_(R,Z)*invB_(R,Z)*bZ_(R,Z); //factor 2 stays under discussion
    }
    double operator()( double R, double Z, double phi)
    {
        return -invB_(R,Z,phi)*invB_(R,Z,phi)*bZ_(R,Z,phi); //factor 2 stays under discussion
    }
    private:    
//     InvB invB_; 
//     BZ bZ_;
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipZ  psipZ_;
    PsipZZ psipZZ_;
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BZ bZ_;    
};
/**
 * @brief K^Z
 */ 
struct CurvatureZ
{
    CurvatureZ( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)),
        psipR_(PsipR(gp.R_0,gp.A,gp.c)),
        psipRR_(PsipRR(gp.R_0,gp.A,gp.c)),
        psipZ_(PsipZ(gp.R_0,gp.A,gp.c)),
        psipRZ_(PsipRZ(gp.R_0,gp.A,gp.c)),
        ipol_(Ipol(gp.R_0,gp.A,psip_)),
        invB_(InvB(gp.R_0,ipol_,psipR_,psipZ_)),
        bR_(BR(gp.R_0,gp.A,psipR_,psipRR_,psipZ_,psipRZ_,invB_)) {
    }
    double operator()( double R, double Z)
    {
        return invB_(R,Z)*invB_(R,Z)*bR_(R,Z); //factor 2 stays under discussion
    }
    double operator()( double R, double Z, double phi)
    {
        return invB_(R,Z,phi)*invB_(R,Z,phi)*bR_(R,Z,phi); //factor 2 stays under discussion
    }
    private:    
//     InvB invB_; 
//     BR bR_;
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipRR  psipRR_;
    PsipZ  psipZ_;    
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BR bR_;   
};

/**
 * @brief nabla_parallel ln(B)
 */ 
struct GradLnB
{
    GradLnB( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)),
        psipR_(PsipR(gp.R_0,gp.A,gp.c)),
        psipRR_(PsipRR(gp.R_0,gp.A,gp.c)),
        psipZ_(PsipZ(gp.R_0,gp.A,gp.c)),
        psipZZ_(PsipZZ(gp.R_0,gp.A,gp.c)),
        psipRZ_(PsipRZ(gp.R_0,gp.A,gp.c)),
        ipol_(Ipol(gp.R_0,gp.A,psip_)),
        invB_(InvB(gp.R_0,ipol_,psipR_,psipZ_)),
        bR_(BR(gp.R_0,gp.A,psipR_,psipRR_,psipZ_,psipRZ_,invB_)), 
        bZ_(BZ(gp.R_0,gp.A,psipR_,psipZ_,psipZZ_,psipRZ_,invB_)) {

    }     
    double operator()( double R, double Z)
    {
       return invB_(R,Z)/R/R*(bR_(R,Z) *psipZ_(R,Z) - bZ_(R,Z)* psipR_(R,Z)) ;
    }
    double operator()( double R, double Z, double phi)
    {
       return invB_(R,Z,phi)/R/R*(bR_(R,Z,phi) *psipZ_(R,Z,phi) - bZ_(R,Z,phi)* psipR_(R,Z,phi)) ;
    }
    private:
//     InvB invB_; 
//     PsipR psipR_;
//     PsipZ psipZ_;
//     BR bR_;
//     BZ bZ_;   
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipRR  psipRR_;
    PsipZ  psipZ_;    
    PsipZZ  psipZZ_;    
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BR bR_;
    BZ bZ_;   
};
/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
struct Field
{
    Field( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)),
        psipR_(PsipR(gp.R_0,gp.A,gp.c)),
        psipRR_(PsipRR(gp.R_0,gp.A,gp.c)),
        psipZ_(PsipZ(gp.R_0,gp.A,gp.c)),
        psipZZ_(PsipZZ(gp.R_0,gp.A,gp.c)),
        psipRZ_(PsipRZ(gp.R_0,gp.A,gp.c)),
        ipol_(Ipol(gp.R_0,gp.A,psip_)),
        invB_(InvB(gp.R_0,ipol_,psipR_,psipZ_)) {
    }
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
//             yp[0][i] =  y[0][i]*psipZ_(y[0][i],y[1][i],y[2][i])/ipol_(y[0][i],y[1][i],y[2][i]);              //dR/dphi =  R/I Psip_Z
//             yp[1][i] = -y[0][i]*psipR_(y[0][i],y[1][i],y[2][i])/ipol_(y[0][i],y[1][i],y[2][i]) ;             //dZ/dphi = -R/I Psip_Z
//             yp[2][i] =  y[0][i]*y[0][i]/invB_(y[0][i],y[1][i],y[2][i])/ipol_(y[0][i],y[1][i],y[2][i]);       //ds/dphi =  R^2 B/I
            yp[2][i] =  y[0][i]*y[0][i]/invB_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i]);       //ds/dphi =  R^2 B/I
            yp[0][i] =  y[0][i]*psipZ_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i]);              //dR/dphi =  R/I Psip_Z
            yp[1][i] = -y[0][i]*psipR_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i]) ;             //dZ/dphi = -R/I Psip_Z
        }
    }
    //inverse B
    double operator()( double R, double Z)
    {
        return invB_(R,Z);
    }
    //inverse B
    double operator()( double R, double Z, double phi)
    {
        return invB_(R,Z,phi);
    }
    
    private:
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipRR psipRR_;
    PsipZ  psipZ_;
    PsipZZ psipZZ_;
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
        
//     Ipol ipol_; 
//     PsipR psipR_;
//     PsipZ psipZ_;
//     InvB invB_;
    
};

/**
 * @brief Sets values to zero outside psipmax and inside psipmin
 */ 
struct Iris
{
    Iris( GeomParameters gp ): 
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax) return 0.;
        if( psip_(R,Z) < gp_.psipmin) return 0.;
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) > gp_.psipmax) return 0.;
        if( psip_(R,Z,phi) < gp_.psipmin) return 0.;
        return 1.;
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
/**
 * @brief Sets values to zero outside psipmax 
 */ 
struct Pupil
{
    Pupil( GeomParameters gp): 
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax) return 0.;
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) > gp_.psipmax) return 0.;
        return 1.;
    }
    private:
    GeomParameters gp_;
    Psip psip_;
};

struct Damping
{
    Damping( GeomParameters gp):
        gp_(gp),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
    double operator( )(double R, double Z)
    {
        if( psip_(R,Z) > gp_.psipmax) return 0.;
        if( psip_(R,Z) < (gp_.psipmax-3.*gp_.alpha)) return 1.;
        return 1. - exp( -( psip_(R,Z)-gp_.psipmax)*( psip_(R,Z)-gp_.psipmax)/2./gp_.alpha/gp_.alpha);
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) > gp_.psipmax) return 0.;
        if( psip_(R,Z,phi) < (gp_.psipmax-3.*gp_.alpha)) return 1.;
        return 1. - exp( -( psip_(R,Z,phi)-gp_.psipmax)*( psip_(R,Z,phi)-gp_.psipmax)/2./gp_.alpha/gp_.alpha);

    }
    private:
    GeomParameters gp_;
    Psip psip_;
};
struct Gradient
{
    Gradient( GeomParameters gp, double lnN_inner):
        gp_(gp),
        lnN_inner_(lnN_inner),
        psip_(Psip(gp.R_0,gp.A,gp.c)) {
        }
   double operator( )(double R, double Z)
    {
        if( psip_(R,Z) < (gp_.psipmin)) return exp(lnN_inner_*log(10)); 
        if( psip_(R,Z) < 0.) return -1./gp_.psipmin*(psip_(R,Z) -gp_.psipmin +exp(lnN_inner_*log(10))*(- psip_(R,Z)));
        return 1.;
    }
    double operator( )(double R, double Z, double phi)
    {
        if( psip_(R,Z,phi) < (gp_.psipmin)) return exp(lnN_inner_*log(10)); 
        if( psip_(R,Z,phi) < 0.) return -1./gp_.psipmin*(psip_(R,Z,phi) -gp_.psipmin +exp(lnN_inner_*log(10))*(- psip_(R,Z,phi)));
        return 1.;
    }
    private:
    GeomParameters gp_;
    double lnN_inner_;
    Psip psip_;
};
struct TestFunction
{
    TestFunction(Psip psip) : psip_(psip){}
    double operator()( double R, double Z, double phi)
    {
        return psip_(R,Z,phi)*sin(phi);
    }
    private:
    Psip psip_;
};

struct DeriTestFunction
{
    DeriTestFunction(Psip psip,PsipR psipR, PsipZ psipZ, Ipol ipol, InvB invB) : psip_(psip), psipR_(psipR), psipZ_(psipZ),ipol_(ipol), invB_(invB) {}
    double operator()( double R, double Z, double phi)
    {
        return ( psip_(R,Z,phi)*ipol_(R,Z,phi)*cos(phi)/R)*invB_(R,Z,phi)/R;
    }
    private:
    Psip psip_;
    PsipR psipR_;
    PsipZ psipZ_;
    Ipol ipol_;
    InvB invB_;
};
} //namespace dg