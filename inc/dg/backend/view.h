#pragma once

#include "tensor_traits.h"

namespace dg
{


/**
 * @brief A vector view class, usable in \c dg::blas1 functions
 *
 * @ingroup view
 * The view class holds a pointer and a size. It does not own the pointer, that
 * is the user is responsible for allocating and deallocating memory.
 * The intention is to use Views in \c dg::blas1 functions.
 *
 * The class can be used as
 * a traditional "view" in the sense that it can view part of a larger contiguous
 * chunk of data and thus apply operations to only part of that data.
 * The second use would be to imitate for example a \c dg::DVec without
 * allocating or copying memory for it. This might be useful if
 * you want to use the \c dg::blas1 functions without specializing \c TensorTraits
 * for your own vector class or deep copying data, like the following example demonstrates:
 * @code
SomeVectorClass vector( 1e6, 20.); //vector of size 1e6, all elements equal 20

//create a view of a device vector to enable parallel execution
dg::View<dg::DVec> view( vector.data(), vector.size());

dg::blas1::copy( 7., view); //elements of vector now equal 7 instead of 20
 * @endcode
 * @tparam SharedContainer \c TensorTraits exists for this class and the \c tensor_category derives from \c SharedVectorTag
 */
template<class SharedContainer >
struct View
{
    ///@brief Initialize empty view
    View(): m_data(nullptr), m_size(0){}

    ///@copydoc construct()
    View( get_pointer_type<SharedContainer> data, unsigned size): m_data(data), m_size(size){ }
    /**
     * @brief Construct view from pointer and size
     *
     * @param data the beginning of the contiguous chunk of data
     * @param size the number of elements in the contiguous chunk of data
     */
    void construct( get_pointer_type<SharedContainer> data, unsigned size)
    {
        m_data = data;
        m_size = size;
    }

    /**
     * @brief Constant Reference of the pointer
     * @return pointer to first element
     */
    const get_pointer_type<SharedContainer>& data() const {
        return m_data;
    }
    /**
     * @brief Write access to the pointer
     * With this function the view can change the data range it views.
     * @return pointer to first element
     */
    get_pointer_type<SharedContainer>& data() {
        return m_data;
    }
    /**
     * @brief Get the size
     * @return number elements in the data view
     */
    unsigned size() const{
        return m_size;
    }
    /**
     * @brief Set the size
     * @return number elements in the data view
     */
    unsigned& size(){
        return m_size;
    }

    /**
     * @brief Swap pointer and size with another View
     * @param src the source view
     */
    void swap( View& src){
        std::swap( src.m_data, m_data);
        std::swap( src.m_size, m_size);
    }
    private:
    get_pointer_type<SharedContainer> m_data;
    unsigned m_size;
};

/**
 * @brief A View has identical value_type and execution_policy as the underlying container
 * @ingroup dispatch
 */
template<class SharedContainer>
struct TensorTraits< View<SharedContainer>>
{
    using value_type = get_value_type<SharedContainer>;
    using tensor_category = SharedVectorTag;
    using execution_policy = get_execution_policy<SharedContainer>;
};

}//namespace dg
