#ifndef _DLT_CUH_
#define _DLT_CUH_

#include <fstream>
#include <stdexcept>
#include <vector>

namespace dg{

/**
 * @brief Struct holding coefficients for Discrete Legendre Transformation (DLT) related operations
 *
 * @tparam T value type
 */
template< class T>
class DLT
{
  public:
      /**
       * @brief Initialize coefficients
       *
       * The constructor reads the data corresponding to given n from the file dlt.dat. 
       * @param n # of polynomial coefficients
       */
    DLT( unsigned n);

    /**
     * @brief Return Gauss-Legendre weights
     *
     * @return weights
     */
    const std::vector<T>& weights()const {return w_;}
    /**
     * @brief Return Gauss-Legendre nodes
     *
     * @return nodes
     */
    const std::vector<T>& abscissas()const {return a_;}
    /**
     * @brief Return forward DLT trafo matrix
     *
     * accesss elements in C-fashion: F_{ij} = forward()[i*n+j]
     * @return forward transformation
     */
    const std::vector<T>& forward()const {return forw_;}
    /**
     * @brief Return backward DLT trafo matrix
     *
     * accesss elements in C-fashion: F_{ij} = backward()[i*n+j]
     * @return backward transformation
     */
    const std::vector<T>& backward()const {return back_;}
    /**
     * @brief Return equidistant backward DLT trafo matrix
     *
     * For vizualisation purposes it is useful to have the values of
     * the DLT - expansion on an equidistant grid.
     * accesss elements in C-fashion: F_{ij} = backwardEQ()[i*n+j]
     * @return equidistant backward transformation
     */
    const std::vector<T>& backwardEQ()const {return backEQ_;}

  private:
    std::vector<T> a_, w_, forw_, back_, backEQ_;
};

template<class T>
DLT<T>::DLT( unsigned n):a_(n), w_(n), forw_(n*n), back_(n*n),backEQ_(n*n)
{
    //get filename
    std::string file( __FILE__);
    file.erase( file.end()-1, file.end());
    file+="dat";
    std::ifstream stream( file.c_str());
    if( stream.fail()) 
        throw "File 'dlt.dat' corrupted or nonexistent!";
    double x;
    for( unsigned i=1; i<n; i++)
    {
        for( unsigned j=0; j<2*i+3*i*i ;j++)
            stream >> x;
    }
    for( unsigned j=0; j<n; j++) stream >> a_[j];
    for( unsigned j=0; j<n; j++) stream >> w_[j];
    for( unsigned j=0; j<n*n; j++) stream >> back_[j];
    for( unsigned j=0; j<n*n; j++) stream >> forw_[j];
    for( unsigned j=0; j<n*n; j++) stream >> backEQ_[j];
    if( !stream.good())
        throw "Error reading file dlt.dat! Is n > 20?";
    stream.close();
}


} //namespace dg
#endif//_DLT_CUH_
