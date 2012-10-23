#ifndef _EXCEPTIONS_
#define _EXCEPTIONS_
#include "message.h"

namespace toefl{

class AllocationError: public  Message
{
  private:
    size_t n, m;
  public:
    AllocationError( size_t n, size_t m, const char *d, const int l):  Message( "Memory couldn't be allocated for: ", d, l), n(n), m(m){}
    void display() const
    {
         Message::display();
        std::cerr << "# of rows " << n << " # of cols "<<m << std::endl;
    }
};

class BadIndex: public  Message
{
  private:
    size_t i, j;
    size_t i_max, j_max;
  public:
    BadIndex( size_t i, size_t i_max, size_t j, size_t j_max, const char *d, const int l):  Message( "Access out of bounds", d, l), i(i), j(j), i_max(i_max), j_max(j_max){}
    void display() const
    {
         Message::display();
        std::cerr << " in row index " << i << " of " <<i_max<< " rows and column index "<<j <<" of "<<j_max<< "columns\n";
    }
};


}


#endif //_EXCEPTIONS_
