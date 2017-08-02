#pragma once

namespace dg
{

//there is probably a better class in boost...
///Helper class to avoid rule of three in grid classes
///Actually it's a very very basic smart pointer that implements 
///a deep copy using the clone() method 
template<class cloneable>
struct Handle
{
    Handle():ptr_(0){}
    /// take ownership of the pointer
    Handle( cloneable* ptr): ptr_(ptr){}
    /// clone given value
    Handle( const cloneable& src): ptr_(src.clone()){}
    Handle( const Handle& src):ptr_(0) {
        if(ptr_!=0) ptr_ = src.ptr->clone(); //deep copy
    }
    Handle& operator=( Handle src) {
        this->swap( src );
        return *this;
    }
    ~Handle(){
        if(ptr_!=0) delete ptr_;
    }
    const cloneable& get()const {return *ptr_;}
    ///takes ownership of the ptr and deletes the current one if non-empty
    void reset( cloneable* ptr){ 
        Handle tmp(ptr);
        *this=tmp;
    }
    ///clones the src
    void reset( const cloneable& src){ 
        Handle tmp(src);
        *this=tmp;
    }
    void swap( Handle& src){
        std::swap(ptr_,src.ptr_);
    }
    private:
    cloneable* ptr_;
};
}
