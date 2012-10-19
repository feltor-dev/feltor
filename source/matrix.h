#ifndef _TL_MATRIX_
#define _TL_MATRIX_

#include <iostream>
#include "fftw3.h"
#include "numerics/fehler.h"

template <class T>
class Matrix;

template <class T>
void permute_cw( Matrix<T>& first, Matrix<T>& second, Matrix<T>& third);

template <class T>
std::ostream& operator<< ( std::ostream& os, const Matrix<T>& mat); 	//Ausgabe der Matrix 		 			cout << setw(5) << a;
template <class T>
std::istream& operator>> ( std::istream& is, Matrix<T>& mat); 


template <class T>
class Matrix
{
  private:
    const size_t n;
    const size_t m;
  protected:
      //maybe an id (static int id) wouldn't be bad to identify in errors
    T **ptr;

  public:
    Matrix( const size_t rows, const size_t cols);
    Matrix( const size_t rows, const size_t cols, const T value);
    ~Matrix();
    Matrix( const Matrix& src);
    Matrix& operator=( const Matrix& src);
    Matrix& operator=( const T value);
    inline void swap( Matrix& rhs);

    T& operator()( const size_t i, const size_t j);
    const T& operator()( const size_t i, const size_t j) const;

    friend void permute_cw<T>( Matrix& first, Matrix& second, Matrix& third);
    friend std::ostream& operator<< <T> ( std::ostream& os, const Matrix& mat); 	//Ausgabe der Matrix 		 			cout << setw(5) << a;
    friend std::istream& operator>><T> ( std::istream& is, Matrix& mat); 
};





#include "matrix.cpp"


#endif //_TL_MATRIX_
