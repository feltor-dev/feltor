#pragma once

#include <any>
#include <memory>
#include <typeindex>

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
    * @note Since the original object \c src is cloned, \c src
    * is allowed to go out of
    * scope after \c ClonePtr was initialized.
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

///@cond
namespace detail
{

/// "A vector whose value type can be changed at runtime"
//should not be public (because of the const behaviour, which is a dirty trick...)
template<template<typename> typename Vector>
struct AnyVector
{
    AnyVector( ) : m_type( typeid( void)){}

    // If not allocated or wrong type; change size and type
    // May need to be called using any_vec.template set<value_type>(size)
    template<class value_type>
    void set(unsigned size)
    {
        auto type_idx = std::type_index( typeid( value_type));
        if( type_idx != m_type)
        {
            m_vec.emplace<Vector<value_type>>(size);
            m_type = type_idx;
        }
        else
        {
            auto ptr = std::any_cast<Vector<value_type>>(
                &m_vec);
            ptr->resize( size);
        }
    }
    // If you think you need this fct. think again, std::vector e.g. will not release
    // memory on resize unless the size is bigger
    //template<class value_type>
    //void set_at_least( unsigned size);
    template<class value_type>
    void swap ( Vector<value_type>& src)
    {
        auto type_idx = std::type_index( typeid( value_type));
        if( type_idx != m_type)
        {
            m_vec.emplace< Vector<value_type>>(std::move(src));
            m_type = type_idx;
        }
        else
        {
            auto& vec = std::any_cast<Vector<value_type>&>( m_vec);
            src.swap(vec);
        }
    }
    // Get read access to underlying buffer
    // May need to be called using any_vec.template get<value_type>()
    template<class value_type>
    const Vector<value_type>& get( ) const
    {
        // throws if not previously set
        return std::any_cast<const Vector<value_type>&>(
            m_vec);
    }
    template<class value_type>
    Vector<value_type>& get( )
    {
        // throws if not previously set
        return std::any_cast<Vector<value_type>&>(
            m_vec);
    }
    private:
    //std::unordered_map< std::type_index, Buffer<std::any>>  m_vec;
    std::any m_vec;
    std::type_index m_type;
};

}//namespace detail
///@endcond

}//namespace dg
