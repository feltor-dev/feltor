#pragma once

namespace dg
{

//there is probably a better class in boost...
/*!@brief a manager class that invokes the clone() method on the managed ptr when copied
*
*When copied invokes a deep copy using the clone() method.
* This class is most useful when a class needs to hold a polymorphic, cloneable oject as a variable. 
@tparam cloneable a type that may be uncopyable/unassignable but provides the clone() method with signature
 - cloneable* clone() const;
@ingroup lowlevel
*/
template<class cloneable>
struct Handle
{
    ///init an empty Handle
    Handle():ptr_(0){}
    /**
    * @brief take ownership of the pointer
    * @param ptr a pointer to object to manage
    */
    Handle( cloneable* ptr): ptr_(ptr){}

    /**
    * @brief clone the given value and manage
    * @param src an object to clone
    */
    Handle( const cloneable& src): ptr_(src.clone()){}
    /**
    * @brief deep copy the given handle
    * @param src an oject to copy, clones the contained object if not empty
    */
    Handle( const Handle& src):ptr_(0) {
        if(src.ptr_!=0) ptr_ = src.ptr_->clone(); //deep copy
    }
    /**
    * @brief deep copy the given handle
    * @param src an oject to copy and swap
    */
    Handle& operator=( Handle src) {
        this->swap( src );
        return *this;
    }
    ///delete managed pointer if not NULL
    ~Handle(){ clear();}

    ///delete managed pointer if not NULL
    void clear(){
        if(ptr_!=0) delete ptr_; 
        ptr_=0;
    }

    /**
    * @brief Get a constant reference to the object on the heap
    * @return a reference to the cloneable object
    * @note undefined if the Handle manages a NULL pointer
    */
    const cloneable& get()const {return *ptr_;}


    /**
    * @brief Non constant access to the object on the heap
    * @return a non-const reference to the cloneable object
    * @note undefined if the Handle manages a NULL pointer
    */
    cloneable& get() {return *ptr_;}

    /**
    * @brief Take the ownership of the given pointer and delete the currently held one if non-empty
    * @param ptr a pointer to an object to manage
    */
    void reset( cloneable* ptr){ 
        Handle tmp(ptr);
        *this=tmp;
    }
    /**
    * @brief Clone the given object and replace the currently held one
    * @param src a cloneable object 
    */
    void reset( const cloneable& src){ 
        Handle tmp(src);
        *this=tmp;
    }
    /**
    * @brief swap the managed pointers
    * @param src another handle
    */
    void swap( Handle& src){
        std::swap(ptr_,src.ptr_);
    }
    private:
    cloneable* ptr_;
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
    ///new T
    Buffer(){
        ptr = new T;
    }
    ///new T(t)
    Buffer( const T& t){
        ptr = new T(t);
    }
    ///delete managed object
    ~Buffer(){
        delete ptr;
    }
    Buffer( const Buffer& src){ 
        ptr = new T(*src.ptr);
    }
    Buffer& operator=( const Buffer& src){
        if( this == &src) return *this;
        Buffer tmp(src);
        std::swap( ptr, tmp.ptr);
        return *this;
    }
    /**
    * @brief Get write access to the data on the heap
    * @return a reference to the data object
    * @attention never try to delete this
    */
    T& data( )const { return *ptr;}
    private:
    T* ptr;
};

}//namespace dg
