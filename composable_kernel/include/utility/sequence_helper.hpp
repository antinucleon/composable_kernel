#ifndef CK_SEQUENCE_HELPER_HPP
#define CK_SEQUENCE_HELPER_HPP

#include "sequence_helper.hpp"

namespace ck {

template <index_t... Is>
__host__ __device__ constexpr auto make_sequence(Number<Is>...)
{
    return Sequence<Is...>{};
}

// F returns index_t
template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence(F, Number<N>)
{
    return typename sequence_gen<N, F>::type{};
}

// F returns Number<>
template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence_v2(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return make_sequence(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

} // namespace ck
#endif
