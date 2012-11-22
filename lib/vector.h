
//!\file Declare the template class Vector
//! \deprecated !!! use std::array in c++0x
/*! \file 
 * \author Matthias Wiesenberger
 * \date 14.10.2011
 */
#ifndef _TL_VECTOR_
#define _TL_VECTOR_

#include <iostream>
#include "message.h"
#include "exceptions.h"
#include <math.h>

 //todo to improve performance review sandersons template lecture
namespace toefl{

//! Numerical vector class to provide a static array of given length
/*!
 * The aim of this class is to provide a static vector class implementation 
 * for numerical applications. Vector is a template class where the template 
 * parameter defines the length of the vector. 
 * \tparam T a numeric type i.e. int, double, complex, etc.
 * \tparam n length of the vector (typically small i.e. 2 or 3)
 *
 */
template <typename T, size_t n>
class Vector
{
  private:
   T vekPtr[n];
  public:
   //! doesn't initialize the elements
   /*!
    * just construct an empty vector of given length
    */
   Vector(){}
   //!initialize the elements to the given value
   /*!
    * \param value the desired value all elements are initialized to
    */
   Vector(const T& value){
	 for(size_t i=0;i<n;i++)
 	   vekPtr[i]=value;
   }
   //!copy every element
   /*!
    * \param src the vector to be copied
    */
   Vector(const Vector& src)
   {
	 for(size_t i=0; i<n; i++)
	   vekPtr[i]=src.vekPtr[i];
   }
   //! assigns a vector 
   /*!
    * every element is assigned seperately
    * \param src the vector to be assigned
    * \return *this
    */
   Vector& operator=(const Vector& src)
   {
	 for(size_t i=0; i<n; i++)
	   vekPtr[i] = src.vekPtr[i];
	 return *this;
   }
   //! assigns a value
   /*!
    * the given value is assigned to every element
    * \param src the value to be assigned
    * \return *this
    */
   Vector& operator=(const T& src)
   {
	 for(size_t i=0; i<n; i++)
	   vekPtr[i] = src;
	 return *this;
   }

   //***********************************************Indexoperatoren**************************************************************************************
   
   //!Index operator
   /*!
    * \param i the index
    * \return a reference to the ith value
    * \attention a range check is performed only if the macro TL_DEBUG is defined
    */
   T& operator[] (const size_t i) throw(BoeserIndex)
   {
   #ifdef TL_DEBUG
     if(i>=n) throw BoeserIndex(i, ping);
   #endif
     return vekPtr[i];
   }
   
   //!constant Index operator
   /*!
    * used if a vector is declared constant
    * \param i the index
    * \return a const reference to the ith value
    * \attention a range check is performed only if the macro TL_DEBUG is defined
    */
   const T& operator[](const size_t i) const throw( BoeserIndex)
   {
   #ifdef TL_DEBUG
     if(i>=n) throw BoeserIndex(i, ping);
   #endif
     return vekPtr[i];
   }
   //************************************************Methoden******************************************************************************************
   /*!
    * \param v the vector to be compared with 
    * \return true if every element of the two vectors are equal false elsewhen
    *
    */
   bool operator ==(const Vector& v) const 
   {
       for(size_t i=0; i<n; i++)
            if(vekPtr[i]!=v[i]) 
                return false;
       return true;
   }
   /*!
    * \param v the vector to be compared with 
    * \return true if at least one element of the two vectors are not equal false elsewhen
    *
    */
   bool operator !=(const Vector& v){return !((*this)==v);}
    
   //! output operator
   /*!
    * output is done in one line, i.e. no newline is added
    * \param os the outstream e.g. cout
    * \param v the Vector to put size_to the outstream
    * \return os the outstream
    * \note the width of the outstream is respected 
    */
   friend std::ostream& operator<< ( std::ostream& os, const Vector& v) 	//Ausgabe des Vectors in einer Zeile		 			cout << setw(5) << a;
   {
     size_t w = os.width();
     for( size_t i=0; i<n; i++)
     {
        os.width(w); 
	    os << v.vekPtr[i]<<" ";	//(Feldbreite gilt immmer nur bis zur nächsten Ausgabe)
     }
     return os;
   }
   //! input operator
   /*!
    *  \param in the instream e.g. cin
    *  \param v the Vector to save the values in
    *  \return the instream
    */
   friend std::istream& operator >> (std::istream& in, Vector& v)
   {
     for(size_t i=0; i<n; i++)
	   in >> v.vekPtr[i];
	 return in;
   }

   //! find position of maximum element
   /*!
    * \return position of greatest element
    * if -3,2,1 are the elements then 1 is returned
    */
   size_t maximum() const				//gibt Position des größtes Element des Vectors zurück							a.max();
   {
     size_t a=0;
     T temp = vekPtr[0];
     for(size_t i=0; i<n-1; i++)
       if(temp < vekPtr[i+1]) temp=vekPtr[i+1], a=i+1;
     return a;
   }
   //! find position of minimum element
   /*!
    * \return position of smallest element
    * if -3,2,1 are the elements then 0 is returned
    */
   size_t minimum() const				//gibt Position des kleinstes Element des Vectors zurück							a.min();
   {
     size_t a=0;
     T temp = vekPtr[0];
     for (size_t i=0; i<n-1; i++)
       if(temp > vekPtr[i+1]) temp=vekPtr[i+1], a=i+1;
     return a;
   }
   //! find position of absolutely maximum element
   /*! 
    * \return position of greates element 
    * if -3,2,1 are the elements then 0 is returned
    */
   size_t abs_maximum() const
   {
      T temp = fabs(vekPtr[0]);
      size_t a = 0;
      for(size_t i=0; i<n-1; i++)
        if(fabs(vekPtr[i+1])>temp) temp = fabs(vekPtr[i+1]), a = i+1;
      return a;
   }
   //! find position of absolutely mimimum element
   /*! 
    * \return position of smallest element 
    * if -3,2,1 are the elements then 2 is returned 
    */
   size_t abs_minimum() const
   {
	 T temp = fabs(vekPtr[0]);
	 size_t a = 0;
	 for(size_t i=0; i<n-1; i++)
	   if(fabs(vekPtr[i+1])<temp) temp = fabs(vekPtr[i+1]), a=i+1;
	 return a;
   }
   
   //! construct a perpendicular vector
   /*!
    * construction is done by turning the vector to the left in the x-z plane
    * \return a new Vector perpendicular to the old
    * \note (a+b).perp()=a.perp()+b.perp(), (a.perp).norm()=a.norm()

    */
   Vector perp() const					//senkrechter Vector in x-z Ebene, Vector wird nach links(!) gedreht  					a.perp() 
   {
       Vector temp;
       for(size_t i=1; i<(n-1); i++)
           temp.vekPtr[i] = (T)0;
       temp.vekPtr[0] = -vekPtr[n-1], temp.vekPtr[n-1] = vekPtr[0];
       return temp;
   }								
   //!the euclidean norm of the vector
   /*!
    * \return the norm of the vector \f$ \left(\sum_{i=0}^{n-1} v_i^2\right)^{1/2}\f$
    */
   T norm() const 					
   {
     T norm = 0;
     for (size_t i=0; i<n; i++)
       norm+=vekPtr[i]*vekPtr[i];
     return sqrt(norm);
   }

   //! scalar division
   /*!
    * \param x a value. If x is 0 an error is thrown
    * \return *this
    */
   Vector& operator/=(T x) 				//skalare Division 							a/=5;
   {
     if(x==0) throw Message("Division durch Null!", ping);
     for(size_t i=0; i<n; i++)
       vekPtr[i]/=x;
     return *this;
   }
   //! scalar divisio
   /*!
    * \param x a value. If x is 0 an error is thrown
    * \return a copy of the result
    */
   Vector operator/ (T x) const throw(Message) 			//skalare Division						a/5;
   {
    if(x==(T)0) throw Message("Division durch Null!", ping);
    Vector temp(n);
    for(size_t i=0; i<n; i++)
      temp[i]= vekPtr[i]/x;
    return temp;
   }  //friend - Funktionen inline definiert, wird allg. gemacht  um implizite Typumwandlungen zu ermöglichen


   //! addition of a vector to the excisting vector
   /*!
    * \param a the vector to be added
    * \return *this
    */
   Vector& operator+=(const Vector& a)							//Addition				a+=b;
   {
     for(size_t i=0; i<n; i++)
       vekPtr[i]+=a.vekPtr[i];
     return *this;
   }
   //! addition of two vectors 
   /*!
    * \param a the first vector to be added
    * \param b the second vector to be added
    * \return a copy of the solution
    */
   friend Vector operator+ (const Vector& a, const Vector& b) throw(Message) 	//Addition zweier Vectoren		a+b;
   {
     Vector temp(n);
     for(size_t i=0; i<n; i++)
       temp.vekPtr[i]=a.vekPtr[i]+b.vekPtr[i];
     return temp;
   }   

   //! addition of a vector and a scalar
   /*!
    * addition is done elementwise
    * \param a the vector to be added
    * \param x the scalar to be added
    * \return a copy of the solution
    */
   friend Vector operator+ (const Vector& a, const T x)				//Addition eines skalaren Werts			a+x;
   {
     Vector temp(n);
     for (size_t i=0; i<n; i++)
       temp.vekPtr[i]=a.vekPtr[i]+x;
     return temp;
   }
   //! addition of a vector and a scalar
   /*!
    * addition is done elementwise
    * \param a the vector to be added
    * \param x the scalar to be added
    * \return a copy of the solution
    */
   friend Vector operator+ (const T x, const Vector& a) {return a+x;}		//						x+a
   
   //! the negation operator
   /*!
    * \return a copy of the negative of *this
    */
   Vector operator-() const
   {
     Vector temp(*this);
     for(size_t i=0; i<n; i++)
       temp[i] = -vekPtr[i];
      return temp;
   }
 
   //! subtraction of two vectors 
   /*!
    * \param a the vector to be subtracted from this 
    * \return *this
    */
   Vector& operator-=(const Vector& a)						//Subtraktion					a-=b;
   {
     for(size_t i=0; i<n;i++)
       vekPtr[i]-=a.vekPtr[i];
     return *this;
   }
   //! subtraction of two vectors 
   /*!
    * \param a the vector to be subtracted from
    * \param b the vector to subtract
    * \return a copy of the solution
    */
   friend Vector operator- (const Vector& a, const Vector& b) throw(Message) //Subtraktion zweier Vectoren		a-b;
   {
     Vector temp(n);
     for(size_t i=0; i<n; i++)
       temp.vekPtr[i]=a.vekPtr[i]-b.vekPtr[i];
     return temp;
   }
   //! subtraction of a vector and a scalar
   /*!
    * \param a the vector to be subtracted from
    * \param x the scalar to subtract
    * \return a copy of the solution
    */
   friend Vector operator- (const Vector& a, const T x)				//Subtraktion eines skalaren Wertes		a-x;
   {
     Vector temp(n);
     for (size_t i=0; i<n; i++)
        temp.vekPtr[i]=a.vekPtr[i]-x;
     return temp;
   }
   //! subtraction of a vector and a scalar
   /*!
    * \param a the vector to subtract
    * \param x the scalar to subtract from
    * \return a copy of the solution
    */
   friend Vector operator- (const T x, const Vector& a)				//						x-a;
   {
     Vector temp(n);
     for (size_t i=0; i<n; i++)
       temp.vekPtr[i]=x-a.vekPtr[i];
     return temp;
   }
   //! scalar multiplication
   /*!
    * \param x the value to multiply with
    * \return *this
    */
   Vector& operator*=(const T x)							//skalare Multiplikation			a*=x
   {
     for(size_t i=0; i<n; i++)
       vekPtr[i]*=x;
     return *this;
   }
   //! scalar multiplication
   /*!
    * \param x the value to multiply with
    * \param a the vector to multiply with
    * \return a copy of the solution
    */
   friend Vector operator* (const T x, const Vector& a) 				//skalare Multiplikation		x*a;
   {
     Vector temp(n);
     for (size_t i=0; i<n; i++)
       temp.vekPtr[i] = x*a.vekPtr[i];
     return temp;
   }
   //! scalar multiplication
   /*!
    * \param a the vector to multiply with
    * \param x the value to multiply with
    * \return a copy of the solution
    */
   friend Vector operator* (const Vector& a, const T x) {return x*a;}		//skalare Multiplikation			a*x;

   //!The scalar product
   /*!
    * \param a the first vector
    * \param b the second vector
    * \return the scalar product \f$ \sum_{i=0}^{n-1} a_ib_i \f$
    */
   friend T operator* (const Vector& a, const Vector& b)			//Skalarprodukt					a*b
   {
     T temp = 0;
     for(size_t i=0; i<n; i++)
       temp+= a.vekPtr[i]*b.vekPtr[i];
     return temp;
   }
   
};
}
#endif //_TL_VECTOR_
