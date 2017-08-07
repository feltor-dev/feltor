/*!
 * \file 
 * \author Matthias Wiesenberger
 * \date 01.08.2017 
 */
#pragma once


#include <exception>
#include <iostream>
#include <sstream>

/*! for the simplified construction of a Message use this Macro*/
#define _ping_ __FILE__, __LINE__ 

namespace dg
{

///@brief small class holding a stringstream 
///@ingroup misc
class Message 
{
  private:
    std::stringstream sstream_;
    Message( const Message&); //we can't copy ostreams in C++
    Message& operator=(const Message&);
  public:
    ///construct an empty message
    Message(){}
    /**
     * @brief Construct message with string
     * @param m puts m into stream
     */
    Message( std::string m){ sstream_<<m;}
    ~Message(){}
    ///@brief add values to the message stream
    /// @note you can't use std::endl or std::flush in here
    template<class T>
    Message & operator << (const T& value)
    {
        sstream_ << value;
        return *this;
    }
    ///return the message contained in the stream as a string
    std::string str() const {return sstream_.str();}
    ///put the sringstream string into the ostream
    ///@note same as os<<m.str();
    friend std::ostream& operator<<(std::ostream& os, const Message& m)
    {
        os<<m.str();
        return os;
    }
};


/*! @brief class intended for the use in throw statements
 *
 * The objects of this class store a message (that describes the error when thrown)
 * that can then be displayed in a catch block
 * \code
 * try{ throw Error(Message()<<"This is error number "<<number, _ping_);}
 * catch( Error& m) {std::cerr << m.what();}
 * \endcode
 * @ingroup misc
 */
class Error : public std::exception
{
  private:
    std::string m;//with a string the Error is copyable
    const char* f;
    const int l;
  public:
     
    /*! @brief Constructor
     *
     * @param message An instance of a Message class
     * @param file The file in which the exception is thrown (contained in the predefined Macro __FILE__)
     * @param line The line in which the exception is thrown (contained in the predefined Macro __LINE__)
     * \note The Macro _ping_ combines __FILE__, __LINE__ in one. 
     */
    Error(const Message& message, const char* file, const int line): f(file), l(line){
        Message tmp;
        tmp << "\n    Message from file **"<<f<<"** in line **" <<l<<"**:\n    "<<message<<"\n";
        m = tmp.str();
    }
    
    /// @return file, line and the message given in the constructor as a string of char
    virtual const char* what() const throw()
    {
        return m.c_str();
    }
    virtual ~Error() throw(){}
};


/**
 * @brief Class you might want to throw in case of a non convergence
 * @ingroup misc
 */
struct Fail : public std::exception
{

    /**
     * @brief Construct from error limit
     *
     * @param eps error not reached
     */
    Fail( double eps): eps( eps) {}
    /**
     * @brief Return error limit
     *
     * @return eps
     */
    double epsilon() const { return eps;}
    /**
     * @brief What string
     *
     * @return string "Failed to converge"
     */
    char const* what() const throw(){ return "Failed to converge";}
    virtual ~Fail() throw(){}
  private:
    double eps;
};

}//namespace dg
