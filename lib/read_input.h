#ifndef _TL_READ_INPUT_
#define _TL_READ_INPUT_

#include <fstream>
#include <sstream>
#include <vector>
#include "../lib/message.h"

namespace toefl{
/*! @brief Extract paramters from a file 
 *
 * A parameter is recognised by the character '=' directly(!) in front of it. 
 * All other characters are ignored!
 * @param filename 
 *   The name of the file to be searched for parameters
 * @return 
 *   A vector containing all parameters read
 *   the zeroth element is -1, the first contains the first parameter
 *   so the size is # of parameters read + 1
 */
std::vector<double> read_input(const char * filename)
{ 
    char s;
    std::vector<double> para; 
    para.push_back( -1.);
    double read;
    std::ifstream is( filename);
    if( is.fail()) //fail() returns true when either eof is reached or failbit or badbit is set
    {
        std::stringstream s;
        s << "Error opening file '" << filename << "'";
        throw Message( s.str().c_str() , ping);
    }
    while(is.good())
    {
        is.get( s);
        if( s == '=') 
        {
            is >> read;
            para.push_back(read); 
            if( is.fail()) //check if read was successful
                throw Message( "Error while reading parameter. Is there a character behind = ?\n", ping);
        }
        if( is.bad()) //check streams integrity
            throw Message( "Error while reading file. File corrupted\n", ping);
    }
    is.close();
    return para;
}
} //namespace toefl

#endif // _TL_READ_INPUT_
