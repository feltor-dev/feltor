#pragma once

namespace dg
{

//there is probably a better class in boost...
/*!@brief a manager class that invokes the \c clone() method on the managed ptr when copied
*
*When copied invokes a deep copy using the \c clone() method.
* This class is most useful when a class needs to hold a polymorphic, cloneable oject as a variable.
@tparam cloneable a type that may be uncopyable/unassignable but provides the \c clone() method with signature
 -  \c cloneable* \c clone() \c const;
@ingroup lowlevel
*/
template<class cloneable>
struct ClonePtr
{
    ///init an empty ClonePtr
    ClonePtr():ptr_(nullptr){}
    /**
    * @brief take ownership of the pointer
    * @param ptr a pointer to object to manage
    */
    ClonePtr( cloneable* ptr): ptr_(ptr){}

    /**
    * @brief clone the given value and manage
    * @param src an object to clone
    */
    ClonePtr( const cloneable& src): ptr_(src.clone()){}

    /**
     * @brief Steal resources (move construct)
     * @param src an object to steal pointer from
     */
    ClonePtr( ClonePtr&& src) : ptr_( src.ptr_) //steal resource
    {
        src.ptr_ = nullptr; //repair
    }
    /**
    * @brief deep copy the given handle using the \c clone() method of \c cloneable
    * @param src an oject to copy, clones the contained object if not empty
    */
    ClonePtr( const ClonePtr& src):ptr_(nullptr) {
        if(src.ptr_!=nullptr) ptr_ = src.ptr_->clone(); //deep copy
    }
    /**
    * @brief deep copy the given handle using the \c clone() method of \c cloneable
    * @param src an oject to copy and swap
    */
    ClonePtr& operator=( ClonePtr src) { //copy and swap, also implements move assignment
        swap( *this, src );
        return *this;
    }
    ///delete managed pointer if not nullptr
    ~ClonePtr(){ clear();}
    /**
    * @brief swap the managed pointers
    * @param first first instance
    * @param second second instance
    */
    friend void swap( ClonePtr& first, ClonePtr& second){ //make std::swap work (ADL)
        using std::swap;
        swap(first.ptr_,second.ptr_);
    }

    ///delete managed pointer if not nullptr
    void clear(){
        if(ptr_!=nullptr) delete ptr_;
        ptr_=nullptr;
    }

    /**
    * @brief Get a constant reference to the object on the heap
    * @return a reference to the cloneable object
    * @note undefined if the ClonePtr manages a nullptr pointer
    */
    const cloneable& get()const {return *ptr_;}


    /**
    * @brief Non constant access to the object on the heap
    * @return a non-const reference to the cloneable object
    * @note undefined if the ClonePtr manages a nullptr pointer
    */
    cloneable& get() {return *ptr_;}

    /**
    * @brief Take the ownership of the given pointer and delete the currently held one if non-empty
    * @param ptr a pointer to an object to manage
    */
    void reset( cloneable* ptr){
        ClonePtr tmp(ptr);
        *this=tmp;
    }
    /**
    * @brief Clone the given object and replace the currently held one
    * @param src a cloneable object
    */
    void reset( const cloneable& src){
        ClonePtr tmp(src);
        *this=tmp;
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
    ///new \c T
    Buffer(){
        ptr = new T;
    }
    ///new \c T(t)
    Buffer( const T& t){
        ptr = new T(t);
    }
    Buffer( T&& t): ptr( t.ptr){ //move (memory steal) construct
        t.ptr = nullptr;
    }
    ///delete managed object
    ~Buffer(){
        delete ptr; //if ptr is nullptr delete does nothing
    }
    Buffer( const Buffer& src){
        ptr = new T(*src.ptr);
    }
    Buffer& operator=( Buffer src){ //copy and swap idiom, also implements move assign
        swap( *this, src);
        return *this;
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
