/*!
 * \file
 * \author Matthias Wiesenberger
 * \date 01.08.2017
 */
#pragma once

#include <exception>
#include <iostream>
#include <sstream>

#define _ping_ __FILE__, __LINE__



/**@file
* @brief Error classes or the dg library
*/
namespace dg
{

///@brief small class holding a stringstream
///@ingroup misc
///\code
///try{ throw Error(Message(_ping_)<<"This is error number "<<number);}
///catch( Error& m) {std::cerr << m.what();}
///\endcode
class Message
{
  private:
    std::stringstream sstream_;
    Message( const Message&); //we can't copy ostreams in C++
    Message& operator=(const Message&);
  public:
    ///construct an empty message
    Message(){}
    /*!@brief Initiate message with the file and line it comes from

     * @param file The file in which the exception is thrown (contained in the predefined Macro <tt> \__FILE__ </tt>)
     * @param line The line in which the exception is thrown (contained in the predefined Macro <tt> \__LINE__ </tt>)
     * \note The Macro <tt> \_ping_ </tt> expands to <tt> \__FILE__, \__LINE__ </tt>_
     */
    Message(const char* file, const int line){
        sstream_ << "\n    Message from file **"<<file<<"** in line **" <<line<<"**:\n    ";
    }
    /**
     * @brief Construct message with string
     * @param m puts m into stream
     */
    Message( std::string m){ sstream_<<m;}
    ~Message(){}
    ///@brief add values to the message stream
    /// @note you can't use \c std::endl or \c std::flush in here
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
 * that can then be displayed in a \c catch block
 * \code
 * try{ throw Error(Message(_ping_)<<"This is error number "<<number);}
 * catch( Error& m) {std::cerr << m.what();}
 * \endcode
 * @ingroup misc
 */
class Error : public std::exception
{
  private:
    std::string m;//with a string the Error is copyable
  public:

    /*! @brief Constructor
     *
     * @param message An instance of a Message class
     */
    Error(const Message& message){
        m = message.str();
    }

    std::string get_message( ) const{return m;}


    ///@brief Appends a message verbatim to the what string
    ///@param message message to append
    void append( const Message& message)
    {
        m+= message.str();
    }

    ///@brief Appends a newline and a message verbatim to the what string
    ///@param message message to append after newline
    void append_line( const Message& message)
    {
        m+= "\n"+message.str();
    }
    /// @return file, line and the message given in the constructor as a string of char
    virtual const char* what() const throw()
    {
        return m.c_str();
    }
    virtual ~Error() throw(){}
};


/**
 * @brief Indicate failure to converge
 * @ingroup misc
 */
struct Fail : public Error
{

    /**
     * @brief Construct from error limit
     *
     * @param eps accuracy not reached
     */
    Fail( double eps): Fail(eps, Message("")){}

    /**
     * @brief Construct from error limit
     *
     * @param eps accuracy not reached
     * @param m additional message
     */
    Fail( double eps, const Message& m): Error(
            Message("\n    FAILED to converge to ")<< eps << "! "<<m),
            eps( eps) {}
    /**
     * @brief Return error limit
     *
     * @return eps
     */
    double epsilon() const { return eps;}
    virtual ~Fail() throw(){}
  private:
    double eps;
};

/**
 * @brief Abort program (both MPI and non-MPI)
 *
 * @code
#ifdef MPI_VERSION
    MPI_Abort(MPI_COMM_WORLD, code);
#endif //WITH_MPI
    exit( code);
 * @endcode
 * @param code The abortion code that will be signalled to the caller (of the program)
 * @ingroup misc
 */
inline void abort_program(int code = -1){
#ifdef MPI_VERSION
    MPI_Abort(MPI_COMM_WORLD, code);
#endif //WITH_MPI
    exit( code);
}


}//namespace dg
