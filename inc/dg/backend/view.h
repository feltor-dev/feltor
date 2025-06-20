#pragma once

#include "tensor_traits.h"

namespace dg
{


/**
 * @brief A vector view class, usable in \c dg functions
 *
 * @ingroup view
 * The view class holds a pointer and a size. It does not own the pointer.
 * The user is responsible for allocating and deallocating memory.
 * The intention is to use Views in \c dg::blas1 functions.
 *
 * The class can be used as
 * a traditional "view" in the sense that it can view part of a larger contiguous
 * chunk of data and thus apply operations to only part of that data.
 * The second use would be to imitate for example a full \c dg::DVec without
 * allocating or copying memory for it. This might be useful if
 * you want to use the \c dg::blas1 functions without specializing \c TensorTraits
 * for your own vector class or deep copying data, like the following example demonstrates:
 * @code
SomeDeviceVectorClass vector( 1e6, 20.); //vector of size 1e6, all elements equal 20

//create a view of a device vector to enable parallel execution
dg::View<dg::DVec> view( vector.data(), vector.size());

dg::blas1::copy( 7., view); //elements of vector now equal 7 instead of 20
 * @endcode
 * @attention when constructing a View from a pointer the user also promises
 * that the pointer can be dereferenced on the device the View acts on.
 * @note Cannot be used as a target \c to in \c dg::construct or \c dg::assign but
 * **can be used as the source** \c from
 *
 * @note You cannot have a \c View of an \c MPI_Vector but you can have an
 * \c MPI_Vector of \c View
 * @tparam ThrustVector \c TensorTraits exists for this class and the
 * \c tensor_category derives from \c ThrustVectorTag
 * @sa \c dg::split
 */
template<class ThrustVector >
struct View
{
    //using iterator = std::conditional_t<std::is_const<ThrustVector>::value,
    //      typename ThrustVector::const_iterator,
    //      typename ThrustVector::iterator>;
    //using const_iterator = typename ThrustVector::const_iterator;
    using pointer = std::conditional_t<std::is_const<ThrustVector>::value,
          typename ThrustVector::const_pointer,
          typename ThrustVector::pointer>;
    using const_pointer = typename ThrustVector::const_pointer;
    using iterator = pointer;
    using const_iterator = const_pointer;
    ///@brief Initialize empty view
    View( void): m_ptr(), m_size(0){}

    /** @brief Construct from another View or Vector
     *
     * The pointer types must be compatible
     * @tparam OtherView Must provide members: \c data() and \c size()
     * @param src Initialize from \c src.data() and \c src.size()
     */
    template<class OtherView>
    View( OtherView& src): m_ptr(src.data()), m_size(src.size()){}

    ///@copydoc construct()
    template<class InputIterator>
    View( InputIterator data, unsigned size): m_ptr(pointer(data)),m_size(size){ }

    /**
     * @brief Construct view from pointer and size
     *
     * @param data the beginning of the contiguous chunk of data
     * @param size the number of elements in the contiguous chunk of data
     * @tparam InputIterator pointer must be constructible from this type
     * @attention when constructing a View from a pointer the user also promises
     * that the pointer can be dereferenced on the device the View acts on.
     */
    template<class InputIterator>
    void construct( InputIterator data, unsigned size)
    {
        m_ptr = pointer(data);
        m_size = size;
    }

    /**
     * @brief Constant Reference of the pointer
     * @return pointer to first element
     */
    pointer data() const {
        return m_ptr;
    }
    /**
     * @brief Iterator to the beginning
     * @return iterator to the first element
     */
    iterator begin() const{
        return iterator(m_ptr);
    }
    /**
     * @brief const Iterator to the beginning
     * @return iterator to the first element
     */
    const_iterator cbegin() const{
        return const_iterator(m_ptr);
    }
    /**
     * @brief Iterator to the end
     * @return iterator to the end
     */
    iterator end() const{
        return iterator(m_ptr + m_size);
    }
    /**
     * @brief const Iterator to the end
     * @return iterator to the end
     */
    const_iterator cend() const{
        return const_iterator(m_ptr + m_size);
    }
    /**
     * @brief Get the size
     * @return number elements in the data view
     */
    unsigned size() const{
        return m_size;
    }

    /**
     * @brief Swap pointer and size with another View
     * @param src the source view
     */
    void swap( View& src){
        std::swap( m_ptr, src.m_ptr);
        std::swap( m_size, src.m_size);
    }
    private:
    pointer m_ptr;
    unsigned m_size;
};

/**
 * @brief A View has identical value_type and execution_policy as the underlying container
 * @ingroup traits
 */
template<class ThrustVector>
struct TensorTraits< View<ThrustVector>>
{
    using value_type = get_value_type<ThrustVector>;
    using tensor_category = ThrustVectorTag;
    using execution_policy = get_execution_policy<ThrustVector>;
};

}//namespace dg
