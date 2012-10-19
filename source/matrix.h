#ifndef _TL_MATRIX_
#define _TL_MATRIX_

#include <iostream>
#include "fftw3.h"

class Matrix
{
  private:
      //maybe an id (static int id) wouldn't be bad to identify in errors
    const size_t n;
    const size_t m;
    double **ptr;

  public:
    Matrix( const size_t rows, const size_t cols);
    Matrix( const size_t rows, const size_t cols, const double value);
    ~Matrix();
    Matrix( const Matrix& src);
    Matrix& operator=( const Matrix& src);
    Matrix& operator=( const double value);
    inline void swap( Matrix& rhs);

    double& operator()( const size_t i, const size_t j);
    const double& operator()( const size_t i, const size_t j) const;


    void dft_lines();


    friend void permute_cw( Matrix& first, Matrix& second, Matrix& third);
    friend std::ostream& operator<< ( std::ostream& os, const Matrix& mat); 	//Ausgabe der Matrix 		 			cout << setw(5) << a;
    friend std::istream& operator>> ( std::istream& is, Matrix& mat); 



};



#endif //_TL_MATRIX_
