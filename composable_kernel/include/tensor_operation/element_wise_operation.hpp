#ifndef CK_ELEMENT_WISE_OPERATION_HPP
#define CK_ELEMENT_WISE_OPERATION_HPP

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        y = x;
    }
};

struct AddHardSwish {
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const
    {

        // hack for Hardswich
        float a = x0 + x1;
        float b = a + 3.f;
        float c = (b > 0.f) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    }

};

struct AddHardSwishAdd {
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1, const T& x2) const
    {


        float a = x0 + x1;
        float b = a + 3.f;
        float c = (b > 0.0f) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        float d = c + x2;
        y       = d;
    }
};
struct AddRelu
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const
    {

        T a = x0 + x1;
        y   = a > 0 ? a : 0;

    }
};

struct AddReluAdd
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1, const T& x2) const
    {

        T a = x0 + x1;
        T b = a > 0 ? a : 0;
        y   = b + x2;

    }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
#endif
