#ifndef _EXCEPTIONS_
#define _EXCEPTIONS_
#include "message.h"

namespace toefl{

/*! @brief Message to be thrown when allocation did not work*/
class AllocationError: public  Message
{
  private:
    size_t n, m;
  public:
    /*! @brief Construct a Message with indices
     *
     * @param n The number of rows
     * @param m The number of columns
     * @param d File name
     * @param l File line
     */
    AllocationError( size_t n, size_t m, const char *d, const int l):  Message( "Memory couldn't be allocated for: ", d, l), n(n), m(m){}
    /*! @brief Display also the number of rows and columns.
     */
    void display() const
    {
         Message::display();
        std::cerr << "# of rows " << n << " # of cols "<<m << std::endl;
    }
};

/*! @brief Message to be thrown when accessing wrong indices in a matrix.*/
class BadIndex: public  Message
{
  private:
    size_t i, j;
    size_t i_max, j_max;
  public:
    /*! @brief Construct a Message with indices
     *
     * @param i The wrong row index
     * @param i_max The number of rows
     * @param j The wrong column index
     * @param j_max The wrong column index
     * @param d File name
     * @param l File line
     */
    BadIndex( size_t i, size_t i_max, size_t j, size_t j_max, const char *d, const int l):  Message( "Access out of bounds", d, l), i(i), j(j), i_max(i_max), j_max(j_max){}
    /*! @brief Display also the indices!
     */
    void display() const
    {
         Message::display();
         std::cerr << " in index (" << i << ", " <<j << ") of "<< i_max<< " rows and  "<<j_max<< " columns\n";
    }
};
///@cond
class BoeserIndex: public Message			
{
  private:
    int index;
  public:
    BoeserIndex( const int i, const char* d, const int l):Message("Bad Index: ",d ,l), index(i) {}	
    void display() const
    {
        Message::display();
        std::cerr << index <<std::endl;
    }
};
///@endcond


}

#endif //_EXCEPTIONS_
