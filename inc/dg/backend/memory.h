#pragma once

#include <memory>

namespace dg
{

//there is probably a better class in boost...
/*!@brief Manager class that invokes the \c clone() method on the managed ptr when copied
*
* When copied invokes a deep copy using the \c clone() method.
* This class is most useful when a class needs to hold a polymorphic, Cloneable oject as a variable.
@tparam Cloneable a type that may be uncopyable/unassignable but provides the \c clone() method with signature
 -  \c Cloneable* \c clone() \c const;
@ingroup lowlevel
*/
template<class Cloneable>
struct ClonePtr
{
    ///init an empty ClonePtr
    ClonePtr( std::nullptr_t value = nullptr):m_ptr(nullptr){}
    /**
    * @brief take ownership of the pointer
    * @param ptr a pointer to object to manage
    */
    ClonePtr( Cloneable* ptr): m_ptr(ptr){}

    /**
    * @brief clone the given value and manage
    * @param src an object to clone
    */
    ClonePtr( const Cloneable& src) : m_ptr( src.clone() ) { }

    /**
    * @brief deep copy the given handle using the \c clone() method of \c Cloneable
    * @param src an oject to copy, clones the contained object if not empty
    */
    ClonePtr( const ClonePtr& src) : m_ptr( src.m_ptr.get() == nullptr ? nullptr : src.m_ptr->clone() ) { }
    /**
    * @brief deep copy the given handle using the \c clone() method of \c Cloneable
    * @param src an oject to copy and swap
    */
    ClonePtr& operator=( const ClonePtr& src)
    {
        //copy and swap
        ClonePtr tmp(src);
        swap( *this, tmp );
        return *this;
    }
    /**
     * @brief Steal resources (move construct)
     * @param src an object to steal pointer from
     */
    ClonePtr( ClonePtr&& src) noexcept : m_ptr( nullptr)
    {
        swap( *this, src); //steal resource
    }
    /**
     * @brief Steal resources (move assignment)
     * @param src an object to steal pointer from
     */
    ClonePtr& operator=( ClonePtr&& src) noexcept
    {
        swap( *this, src );
        return *this;
    }
    /**
    * @brief swap the managed pointers
    *
    * This follows the discussion in
    * https://stackoverflow.com/questions/5695548/public-friend-swap-member-function
    * @note the std library does call this function via unqualified calls in
    * many algorithms, for example in std::iter_swap, it is just std::swap that
    * will not call it directly. Most of the time this is not an issue because
    * we have move assignments now, but if you do want to enable free swap
    * functions like these use:
    * @code
    * using std::swap;
    * swap(a,b);
    * @endcode
    * @param first first instance
    * @param second second instance
    */
    friend void swap( ClonePtr& first, ClonePtr& second)
    {
        std::swap(first.m_ptr,second.m_ptr);
    }

    /**
    * @brief Replace the managed object
    *
    * Take the ownership of the given pointer and delete the currently
    * held one if non-empty
    * @param ptr a pointer to a new object to manage
    */
    void reset( Cloneable* ptr){
        m_ptr.reset( ptr);
    }
    ///Releases ownership of managed object, \c get() returns \c nullptr after call
    Cloneable* release() noexcept { m_ptr.release();}
    /**
    * @brief Clone the given object and replace the currently held one
    * @param src a Cloneable object
    */
    void reset( const Cloneable& src){
        ClonePtr tmp(src);
        swap(*this, tmp);
    }

    /**
    * @brief Get a pointer to the object on the heap
    * @return a pointer to the Cloneable object or \c nullptr if no object owned
    */
    Cloneable * get() {return m_ptr.get();}
    /**
    * @brief Get a constant pointer to the object on the heap
    * @return a pointer to the Cloneable object or \c nullptr if no object owned
    */
    const Cloneable* get() const {return m_ptr.get();}

    ///Dereference pointer to owned object, i.e. \c *get()
    Cloneable& operator*() { return *m_ptr;}
    ///Dereference pointer to owned object, i.e. \c *get()
    const Cloneable& operator*() const { return *m_ptr;}
    ///Dereference pointer to owned object, i.e. \c get()
    Cloneable* operator->() { return m_ptr.operator->();}
    ///Dereference pointer to owned object, i.e. \c get()
    const Cloneable* operator->()const { return m_ptr.operator->();}
    ///\c true if \c *this owns an object, \c false else
    explicit operator bool() const{ return (bool)m_ptr;}


    private:
    std::unique_ptr<Cloneable> m_ptr;
};

//Memory buffer class: data can be written even if the object is const
/**
* @brief a manager class that invokes the copy constructor on the managed ptr when copied (deep copy)
*
* this class is most useful as a memory buffer for classes that need
* some workspace to fulfill their task but do otherwise not change their state. A buffer object
can be declared const while the data it holds are still writeable.
* @tparam T must be default constructible and copyable
* @ingroup lowlevel
*/
template< class T>
struct Buffer
{
    ///new \c T
    Buffer(){
        ptr = new T;
    }
    ///new \c T(t)
    Buffer( const T& t){
        ptr = new T(t);
    }
    Buffer( const Buffer& src){ //copy
        ptr = new T(*src.ptr);
    }
    Buffer( Buffer&& t): ptr( t.ptr){ //move (memory steal) construct
        t.ptr = nullptr;
    }
    Buffer& operator=( Buffer src){ //copy and swap idiom, also implements move assign
        swap( *this, src);
        return *this;
    }
    ///delete managed object
    ~Buffer(){
        delete ptr; //if ptr is nullptr delete does nothing
    }
    friend void swap( Buffer& first, Buffer& second) //make std::swap work (ADL)
    {
        using std::swap;
        swap( first.ptr, second.ptr);
    }


    /**
    * @brief Get write access to the data on the heap
    * @return a reference to the data object
    * @attention never try to delete the returned reference
    */
    T& data( )const { return *ptr;}

    private:
    T* ptr;
};

}//namespace dg
