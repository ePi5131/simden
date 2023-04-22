// simden.hpp
// (C) 2023 ePi
// Released under The 1-Clause BSD License, see LICENSE.

#pragma once
#include <array>
#include <bit>
#include <utility>
#include <compare>
#include <concepts>
#include <tuple>
#include <functional>
#include <ranges>
#include <algorithm>

#include <intrin.h>

namespace simden {

    using m128f   =  __m128;
    using m128i  =  __m128i;
    using m128d  =  __m128d;
    using m128h  =  __m128h;
    using m128bh =  __m128bh;
    using m256f   =  __m256;
    using m256i  =  __m256i;
    using m256d  =  __m256d;
    using m256h  =  __m256h;
    using m256bh =  __m256bh;
    using m512f   =  __m512;
    using m512i  =  __m512i;
    using m512d  =  __m512d;
    using m512h  =  __m512h;
    using m512bh =  __m512bh;
    
    template<class T = int>
    [[nodiscard]] consteval T make_flag(int scale, auto... values) {
        auto impl = [](this const auto& self, auto scale, auto head, auto... tail) {
            if constexpr (sizeof...(tail) == 0) {
                return head;
            }
            else {
                if (std::cmp_greater_equal(head, 1u << scale)) throw "flag elm is too big";

                return head | (self(scale, tail...) << scale);
            }
        };
        return static_cast<T>(impl(scale, values...));
    }

    enum class intrinsic_type : size_t {
        itSSE,
        itSSE2,
        itSSE3,
        itSSSE3,
        itSSE41,
        itSSE42,
        itAVX,
        itFMA,
        itAVX2,
        itAVX512F,
        itAVX512DQ,
        itAVX512BW,
        itAVX512VL,

        _size,
    };
    using enum intrinsic_type;

    namespace { namespace detail {
        using intrinsics_flag_base = std::array<bool, std::to_underlying(intrinsic_type::_size)>;

        struct intrinsics_tmp_dummy { intrinsics_tmp_dummy() = default; };

        template<class T>
        concept m128bitreg =
            std::same_as<T, m128f> ||
            std::same_as<T, m128i> ||
            std::same_as<T, m128d> ||
            std::same_as<T, m128h> ||
            std::same_as<T, m128bh>;

        template<class T>
        concept m256bitreg =
            std::same_as<T, m256f> ||
            std::same_as<T, m256i> ||
            std::same_as<T, m256d> ||
            std::same_as<T, m256h> ||
            std::same_as<T, m256bh>;

        template<class T>
        concept m512bitreg =
            std::same_as<T, m512f> ||
            std::same_as<T, m512i> ||
            std::same_as<T, m512d> ||
            std::same_as<T, m512h> ||
            std::same_as<T, m512bh>;

        template<class T>
        concept simd_reg = m128bitreg<T> || m256bitreg<T> || m512bitreg<T>;

        template<class T>
        concept simd_mask =
            std::same_as<T, __mmask8> ||
            std::same_as<T, __mmask16> ||
            std::same_as<T, __mmask32> ||
            std::same_as<T, __mmask64>;

        template<class T>
        concept simd_vec = requires(const T t) {
            typename T::elm_type;
            typename T::reg_type;
            { T::elm_count } -> std::convertible_to<size_t>;
            { T::elm_size } -> std::convertible_to<size_t>;
            { t } -> std::convertible_to<typename T::reg_type>;
        };

        inline auto to_reg(simd_reg auto r) -> decltype(r) {
            return r;
        }

        template<simd_vec V>
        auto to_reg(V v) -> V::reg_type {
            return v;
        }

        template<class U, class V>
        concept is_same_reg = requires(U u, V v) {
            requires std::same_as<
                decltype(to_reg(u)),
                decltype(to_reg(v))
            >;
        };
        
        template<std::same_as<m128i> To> To mmcast_impl(const m128f& x) { return _mm_castps_si128      (x); }
        template<std::same_as<m128d> To> To mmcast_impl(const m128f& x) { return _mm_castps_pd         (x); }
        template<std::same_as<m256f> To> To mmcast_impl(const m128f& x) { return _mm256_castps128_ps256(x); }
        template<std::same_as<m512f> To> To mmcast_impl(const m128f& x) { return _mm512_castps128_ps512(x); }
        
        template<std::same_as<m128f> To> To mmcast_impl(const m128i& x) { return _mm_castsi128_ps      (x); }
        template<std::same_as<m128d> To> To mmcast_impl(const m128i& x) { return _mm_castsi128_pd      (x); }
        template<std::same_as<m256i> To> To mmcast_impl(const m128i& x) { return _mm256_castsi128_si256(x); }
        template<std::same_as<m512i> To> To mmcast_impl(const m128i& x) { return _mm512_castsi128_si512(x); }
        
        template<std::same_as<m128f> To> To mmcast_impl(const m128d& x) { return _mm_castpd_ps         (x); }
        template<std::same_as<m128i> To> To mmcast_impl(const m128d& x) { return _mm_castpd_si128      (x); }
        template<std::same_as<m256d> To> To mmcast_impl(const m128d& x) { return _mm256_castpd128_pd256(x); }
        template<std::same_as<m512d> To> To mmcast_impl(const m128d& x) { return _mm512_castpd128_pd512(x); }


        template<std::same_as<m256i> To> To mmcast_impl(const m256f& x) { return _mm256_castps_si256   (x); }
        template<std::same_as<m256d> To> To mmcast_impl(const m256f& x) { return _mm256_castps_pd      (x); }
        template<std::same_as<m128f> To> To mmcast_impl(const m256f& x) { return _mm256_castps256_ps128(x); }
        template<std::same_as<m512f> To> To mmcast_impl(const m256f& x) { return _mm512_castps256_ps512(x); }
        
        template<std::same_as<m256f> To> To mmcast_impl(const m256i& x) { return _mm256_castsi256_ps   (x); }
        template<std::same_as<m256d> To> To mmcast_impl(const m256i& x) { return _mm256_castsi256_pd   (x); }
        template<std::same_as<m128i> To> To mmcast_impl(const m256i& x) { return _mm256_castsi256_si128(x); }
        template<std::same_as<m512i> To> To mmcast_impl(const m256i& x) { return _mm512_castsi256_si512(x); }
        
        template<std::same_as<m256f> To> To mmcast_impl(const m256d& x) { return _mm256_castpd_ps      (x); }
        template<std::same_as<m256i> To> To mmcast_impl(const m256d& x) { return _mm256_castpd_si256   (x); }
        template<std::same_as<m128d> To> To mmcast_impl(const m256d& x) { return _mm256_castpd256_pd128(x); }
        template<std::same_as<m512d> To> To mmcast_impl(const m256d& x) { return _mm512_castpd256_pd512(x); }


        template<std::same_as<m512i> To> To mmcast_impl(const m512f& x) { return _mm512_castps_si512   (x); }
        template<std::same_as<m512d> To> To mmcast_impl(const m512f& x) { return _mm512_castps_pd      (x); }
        template<std::same_as<m128f> To> To mmcast_impl(const m512f& x) { return _mm512_castps512_ps128(x); }
        template<std::same_as<m256f> To> To mmcast_impl(const m512f& x) { return _mm512_castps512_ps256(x); }
        
        template<std::same_as<m512f> To> To mmcast_impl(const m512i& x) { return _mm512_castsi512_ps   (x); }
        template<std::same_as<m512d> To> To mmcast_impl(const m512i& x) { return _mm512_castsi512_pd   (x); }
        template<std::same_as<m128i> To> To mmcast_impl(const m512i& x) { return _mm512_castsi512_si128(x); }
        template<std::same_as<m256i> To> To mmcast_impl(const m512i& x) { return _mm512_castsi512_si256(x); }
        
        template<std::same_as<m512f> To> To mmcast_impl(const m512d& x) { return _mm512_castpd_ps      (x); }
        template<std::same_as<m512i> To> To mmcast_impl(const m512d& x) { return _mm512_castpd_si512   (x); }
        template<std::same_as<m128d> To> To mmcast_impl(const m512d& x) { return _mm512_castpd512_pd128(x); }
        template<std::same_as<m256d> To> To mmcast_impl(const m512d& x) { return _mm512_castpd512_pd256(x); }

        template<class T, class To>
        concept mm_direct_castable = requires(T t) {
            mmcast_impl<To>(t);
        };

        template<std::same_as<m128f> To, mm_direct_castable<m256f> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m256f>(x)); }
        template<std::same_as<m128i> To, mm_direct_castable<m256i> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m256i>(x)); }
        template<std::same_as<m128d> To, mm_direct_castable<m256d> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m256d>(x)); }
        template<std::same_as<m128f> To, mm_direct_castable<m512f> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m512f>(x)); }
        template<std::same_as<m128i> To, mm_direct_castable<m512i> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m512i>(x)); }
        template<std::same_as<m128d> To, mm_direct_castable<m512d> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m512d>(x)); }
        
        template<std::same_as<m256f> To, mm_direct_castable<m128f> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m128f>(x)); }
        template<std::same_as<m256i> To, mm_direct_castable<m128i> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m128i>(x)); }
        template<std::same_as<m256d> To, mm_direct_castable<m128d> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m128d>(x)); }
        template<std::same_as<m256f> To, mm_direct_castable<m512f> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m512f>(x)); }
        template<std::same_as<m256i> To, mm_direct_castable<m512i> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m512i>(x)); }
        template<std::same_as<m256d> To, mm_direct_castable<m512d> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m512d>(x)); }
        
        template<std::same_as<m512f> To, mm_direct_castable<m128f> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m128f>(x)); }
        template<std::same_as<m512i> To, mm_direct_castable<m128i> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m128i>(x)); }
        template<std::same_as<m512d> To, mm_direct_castable<m128d> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m128d>(x)); }
        template<std::same_as<m512f> To, mm_direct_castable<m256f> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m256f>(x)); }
        template<std::same_as<m512i> To, mm_direct_castable<m256i> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m256i>(x)); }
        template<std::same_as<m512d> To, mm_direct_castable<m256d> From> To mmcast_impl2(const From& x) { return mmcast_impl<To>(mmcast_impl<m256d>(x)); }

        template<simd_reg To, simd_reg From> [[nodiscard]] To mmcast(const From& x) noexcept {
            if constexpr (std::same_as<To, From>) {
                return x;
            }
            else if constexpr (mm_direct_castable<To, From>) {
                return mmcast_impl<To>(x);
            }
            else if constexpr (requires(const From from){
                { mmcast_impl2<To>(from) } -> std::convertible_to<To>;
            }) {
                return mmcast_impl2<To>(x);
            }
            else {
                throw "おい";
            }
        }

        namespace emulate {
            template<simd_vec T, class F>
            [[nodiscard]] auto simd_2op(const T& x, const T& y, F&& f) {
                using reg = T::reg_type;

                const auto& mx = to_reg(x);
                const auto& my = to_reg(y);
                reg ret;
                for (size_t i = 0; i < T::elm_count; i++) {
                    ret[i] = std::invoke_r<typename T::elm_type>(f, mx[i], my[i]);
                }
                return ret;
            }

            template<simd_vec T>
            [[nodiscard]] constexpr auto set1(typename T::elm_type x) noexcept {
                typename T::reg_type m;
                for (size_t i = 0; i < T::elm_count; i++)
                    std::memcpy(&m[i], &x, T::elm_size);
                return m;
            }

            template<simd_vec To, simd_vec From>
            [[nodiscard]] constexpr To cast(const From& x) noexcept {
                constexpr auto to_n = std::tuple_size_v<typename To::reg_type>;
                constexpr auto from_n = std::tuple_size_v<typename From::reg_type>;
                if constexpr (to_n == from_n) {
                    return x;
                }
                else if constexpr (to_n < from_n) {
                    typename To::reg_type m;
                    const From::reg_type& from_m = x;
                    for (size_t i = 0; i < to_n; i++) {
                        m[i] = from_m[i];
                    }
                    return m;
                }
                else {
                    typename To::reg_type m;
                    const From::reg_type& from_m = x;
                    for (size_t i = 0; i < from_n; i++) {
                        m[i] = from_m[i];
                    }
                    for (size_t i = from_n; i < to_n; i++) {
                        m[i] = 0;
                    }
                    return m;
                }
            }

        }

    }}

    using detail::simd_vec;
    using detail::simd_reg;
    using detail::mmcast;

    struct intrinsics_flag : public detail::intrinsics_flag_base {
        template<std::same_as<intrinsic_type>... T>
        [[nodiscard]] constexpr intrinsics_flag(T... v) : detail::intrinsics_flag_base{} {
            ((this->at(std::to_underlying(v)) = true), ...);
        }

        [[nodiscard]] constexpr bool has_flag(intrinsic_type t) const {
            return at(std::to_underlying(t));
        }

        [[nodiscard]] constexpr void set(intrinsic_type t, bool value = true) {
            at(std::to_underlying(t)) = value;
        }
    };

    [[nodiscard]]
    inline intrinsics_flag get_intrinsics_flag() noexcept {
        static intrinsics_flag instance = []{
            intrinsics_flag ret{};

            int cpuinfo[4];
            __cpuid(cpuinfo, 0);
            auto basic_n = cpuinfo[0];

            if (basic_n < 1) return ret;

            __cpuid(cpuinfo, 1);
            if (cpuinfo[3] & (1u<<25)) ret.set(itSSE);
            if (cpuinfo[3] & (1u<<26)) ret.set(itSSE2);

            if (cpuinfo[2] & (1u<< 0)) ret.set(itSSE3);
            if (cpuinfo[2] & (1u<< 9)) ret.set(itSSSE3);
            if (cpuinfo[2] & (1u<<19)) ret.set(itSSE41);
            if (cpuinfo[2] & (1u<<20)) ret.set(itSSE42);
            if (cpuinfo[2] & (1u<<28)) ret.set(itAVX);
            if (cpuinfo[2] & (1u<<12)) ret.set(itFMA);

            if (basic_n < 7) return ret;

            __cpuidex(cpuinfo, 7, 0);
            if (cpuinfo[1] & (1u<< 5)) ret.set(itAVX2);
            if (cpuinfo[1] & (1u<<16)) ret.set(itAVX512F);
            if (cpuinfo[1] & (1u<<17)) ret.set(itAVX512DQ);
            if (cpuinfo[1] & (1u<<30)) ret.set(itAVX512BW);
            if (cpuinfo[1] & (1u<<31)) ret.set(itAVX512VL);

            return ret;
        }();
        return instance;
    }

    template<intrinsics_flag flag>
    class intrinsics {
        using dummy = detail::intrinsics_tmp_dummy;

    public:
        [[nodiscard]]
        constexpr static bool is_supported(const intrinsics_flag& x) noexcept {
            for (size_t i = 0; i < x.size(); i++) {
                if (!flag[i]) continue;
                if (!x[i]) return false;
            }
            return true;
        }

        struct alignas(m128f) f32x4 {
            using elm_type = float;
            static constexpr size_t elm_size = 4;
            static constexpr size_t elm_count = 4;

            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m128f;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            constexpr f32x4() noexcept requires (flag.has_flag(itSSE)) {};

            [[nodiscard]]
            constexpr f32x4(const reg_type& m) noexcept requires (flag.has_flag(itSSE)) : m{ m } {}

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr f32x4(T... t) noexcept requires (flag.has_flag(itSSE)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{ .m128_f32{t...} }
                #else
                m{t...}
                #endif
            {}

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            template<bool Aligned = false>
            [[nodiscard]]
            inline static f32x4 load(const float* p) noexcept requires (flag.has_flag(itSSE)) {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                if constexpr (Aligned) {
                    return _mm_load_ps(p);
                }
                else {
                    return _mm_loadu_ps(p);
                }
                #else
                reg_type ret;
                for (size_t i = 0; i < elm_count; i++) {
                    ret[i] = p[i];
                }
                return ret;
                #endif
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<f32x4>);
        
        struct alignas(m128i) i32x4 {
            using elm_type = int;
            static constexpr size_t elm_size = 4;
            static constexpr size_t elm_count = 4;
            
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m128i;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            constexpr i32x4() noexcept requires (flag.has_flag(itSSE)) {};

            [[nodiscard]]
            constexpr i32x4(const reg_type& m) noexcept requires (flag.has_flag(itSSE)) : m{ m } {}

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr i32x4(T... t) noexcept requires (flag.has_flag(itSSE)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{ .m128i_i32{t...} }
                #else
                m{t...}
                #endif
            {}

            template<bool Aligned = false>
            [[nodiscard]]
            inline static i32x4 load(const int* p) noexcept requires (flag.has_flag(itSSE)) {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                if constexpr (Aligned) {
                    return _mm_load_si128(reinterpret_cast<const m128i*>(p));
                }
                else {
                    return _mm_loadu_si128(reinterpret_cast<const m128i*>(p));
                }
                #else
                reg_type ret;
                for (size_t i = 0; i < elm_count; i++) {
                    ret[i] = p[i];
                }
                return ret;
                #endif
            }

            [[nodiscard]]
            friend inline i32x4 operator+(const i32x4& x, const i32x4& y) noexcept {
                return add(x, y);
            }

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            template<bool Aligned = false>
            [[nodiscard]]
            inline void store(std::array<int, elm_count>& x) const noexcept {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                if constexpr (Aligned) {
                    _mm_store_si128(reinterpret_cast<m128i*>(x.data()), m);
                }
                else {
                    _mm_storeu_si128(reinterpret_cast<m128i*>(x.data()), m);
                }
                #else
                x = m;
                #endif
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<i32x4>);

        struct alignas(m128i) u8x4 {
            using elm_type = unsigned char;
            static constexpr size_t elm_size = 1;
            inline static constexpr size_t elm_count = 4;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m128i;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            [[nodiscard]]
            constexpr u8x4() noexcept requires (flag.has_flag(itSSE)) {}

            [[nodiscard]]
            constexpr u8x4(const reg_type& m) noexcept requires (flag.has_flag(itSSE)) : m{ m } {}

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr u8x4(T... t) noexcept requires (flag.has_flag(itSSE)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{ .m128i_u8{t...} }
                #else
                m{t...}
                #endif
            {}

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            [[nodiscard]]
            inline static u8x4 load(const void* p) noexcept requires (flag.has_flag(itSSE)) {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                return _mm_loadu_si32(p);
                #else
                u8x4::reg_type ret;
                std::memcpy(&ret, p, 4);
                return ret;
                #endif
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<u8x4>);

        struct alignas(m128i) u8x8 {
            using elm_type = unsigned char;
            static constexpr size_t elm_size = 1;
            inline static constexpr size_t elm_count = 8;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m128i;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            [[nodiscard]]
            constexpr u8x8() noexcept requires (flag.has_flag(itSSE)) {}

            [[nodiscard]]
            constexpr u8x8(const reg_type& m) noexcept requires (flag.has_flag(itSSE)) : m{ m } {}

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr u8x8(T... t) noexcept requires (flag.has_flag(itSSE)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{ .m128i_u8{t...} }
                #else
                m{t...}
                #endif
            {}

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            [[nodiscard]]
            inline static u8x8 load(const void* p) noexcept requires (flag.has_flag(itSSE)) {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                return _mm_loadu_si64(p);
                #else
                u8x8::reg_type ret;
                std::memcpy(&ret, p, 8);
                return ret;
                #endif
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<u8x8>);
        
        struct alignas(m128i) u8x16 {
            using elm_type = unsigned char;
            static constexpr size_t elm_size = 1;
            inline static constexpr size_t elm_count = 16;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m128i;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            constexpr u8x16() noexcept requires (flag.has_flag(itSSE)) {}

            [[nodiscard]]
            constexpr u8x16(const reg_type& m) noexcept requires (flag.has_flag(itSSE)) : m{ m } {}

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr u8x16(T... t) noexcept requires (flag.has_flag(itSSE)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{ .m128i_u8{t...} }
                #else
                m{t...}
                #endif
            {}

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            template<bool Aligned = false>
            [[nodiscard]]
            inline static u8x16 load(const void* p) noexcept requires (flag.has_flag(itSSE)) {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                if constexpr (Aligned) {
                    return _mm_load_si128(static_cast<const m128i*>(p));
                }
                else {
                    return _mm_loadu_si128(static_cast<const m128i*>(p));
                }
                #else
                u8x16::reg_type ret;
                std::memcpy(&ret, p, 16);
                return ret;
                #endif
            }

            template<bool Aligned = false>
            [[nodiscard]]
            inline void store(void* ptr) const noexcept requires (flag.has_flag(itSSE)) {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                if constexpr (Aligned) {
                    _mm_store_si128(reinterpret_cast<m128i*>(ptr), m);
                }
                else {
                    _mm_storeu_si128(reinterpret_cast<m128i*>(ptr), m);
                }
                #else
                std::memcpy(ptr, &m, elm_size * elm_count);
                #endif
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<u8x16>);

        struct alignas(m256i) i32x8 {
            using elm_type = int;
            static constexpr size_t elm_size = 4;
            static constexpr size_t elm_count = 8;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m256i;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            constexpr i32x8() noexcept requires (flag.has_flag(itAVX)) {}

            [[nodiscard]]
            constexpr i32x8(const reg_type& m) noexcept requires (flag.has_flag(itAVX)) : m{ m } {}

            [[nodiscard]]
            constexpr i32x8(const i32x4& m0, const i32x4& m1) noexcept requires (flag.has_flag(itAVX)) :
            #ifndef SIMDEN_EMULATE_INTRINSICS
                m{ _mm256_set_m128i(m1, m0) }
            #else
                //m{ std::make_from_tuple<reg_type>(std::tuple_cat(m0.m,m1.m)) }
                m{
                    m0.m[0],
                    m0.m[1],
                    m0.m[2],
                    m0.m[3],
                    m1.m[0],
                    m1.m[1],
                    m1.m[2],
                    m1.m[3],
                }
            #endif
            {}
            
            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr i32x8(T... t) noexcept requires (flag.has_flag(itAVX)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{.m256i_i32{t...}}
                #else
                m{t...}
                #endif
            {}
            
            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            [[nodiscard]]
            friend inline i32x8 operator+(const i32x8& x, const i32x8& y) noexcept {
                return add(x, y);
            }

            [[nodiscard]]
            friend inline i32x8 operator*(const i32x8& x, const i32x8& y) noexcept {
                return mul(x, y);
            }

            [[nodiscard]]
            friend inline i32x8 operator&(const i32x8& x, const i32x8& y) noexcept {
                return band(x, y);
            }

            [[nodiscard]]
            friend inline i32x8 operator|(const i32x8& x, const i32x8& y) noexcept {
                return bor(x, y);
            }

            template<bool Aligned = false>
            [[nodiscard]]
            inline void store(std::array<int, elm_count>& x) const noexcept {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                if constexpr (Aligned) {
                    _mm256_store_si256(reinterpret_cast<m256i*>(x.data()), m);
                }
                else {
                    _mm256_storeu_si256(reinterpret_cast<m256i*>(x.data()), m);
                }
                #else
                x = m;
                #endif
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<i32x8>);

        struct alignas(m256f) f32x8 {
            using elm_type = float;
            static constexpr size_t elm_size = 4;
            static constexpr size_t elm_count = 8;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m256f;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            [[nodiscard]]
            constexpr f32x8() noexcept requires (flag.has_flag(itAVX)) {}

            [[nodiscard]]
            constexpr f32x8(const reg_type& m) noexcept requires (flag.has_flag(itAVX)) : m{ m } {}

            [[nodiscard]]
            constexpr f32x8(const f32x4& m0, const f32x4& m1) noexcept requires (flag.has_flag(itAVX)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{_mm256_set_m128(m1, m0)}
                #else
                //m{std::make_from_tuple<reg_type>(std::tuple_cat(m0.m,m1.m))}
                m{
                    m0.m[0],
                    m0.m[1],
                    m0.m[2],
                    m0.m[3],
                    m1.m[0],
                    m1.m[1],
                    m1.m[2],
                    m1.m[3],
                }
                #endif
            {}

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr f32x8(T... t) noexcept requires (flag.has_flag(itAVX)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{.m256_f32{t...}}
                #else
                m{t...}
                #endif
            {}

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            [[nodiscard]]
            friend inline f32x8 operator+(const f32x8& x, const f32x8& y) noexcept {
                return add(x, y);
            }

            [[nodiscard]]
            friend inline f32x8 operator-(const f32x8& x, const f32x8& y) noexcept {
                return sub(x, y);
            }

            [[nodiscard]]
            friend inline f32x8 operator*(const f32x8& x, const f32x8& y) noexcept {
                return mul(x, y);
            }

            [[nodiscard]]
            friend inline f32x8 operator>=(const f32x8& x, const f32x8& y) noexcept {
                return cmp_ge(x, y);
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<f32x8>);

        struct mask16 {
            using elm_type = bool;
            static constexpr size_t elm_count = 16;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = __mmask16;
            #else
            using reg_type = std::array<bool, elm_count>;
            #endif

            [[nodiscard]]
            constexpr mask16(const reg_type& m) noexcept requires(flag.has_flag(itAVX512F)): m{m} {}
            #ifdef SIMDEN_EMULATE_INTRINSICS
            [[nodiscard]]
            constexpr mask16(std::integral auto x) noexcept requires (flag.has_flag(itAVX512F)) :
                m{ [x] {
                    reg_type ret;
                    for (size_t i = 0; i < elm_count; i++) {
                        ret[i] = (x >> i) & 1;
                    }
                    return ret;
                }() }
            {}
            #endif

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr mask16(T... f) noexcept requires(flag.has_flag(itAVX512F)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{make_flag<reg_type>(1, f...)}
                #else
                m{static_cast<bool>(f)...}
                #endif
            {}

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            [[nodiscard]]
            constexpr uint16_t to_int() const noexcept {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                return m;
                #else
                uint16_t ret = 0;
                for (size_t i = 0; i < elm_count; i++) {
                    ret |= m[i] << i;
                }
                return ret;
                #endif
            }

            [[nodiscard]]
            constexpr bool operator[](size_t i) const noexcept {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                return (m >> i) & 1;
                #else
                return m[i];
                #endif
            }

        private:
            reg_type m;
        };

        struct alignas(m512i) i32x16 {
            using elm_type = int;
            static constexpr size_t elm_size = 4;
            static constexpr size_t elm_count = 16;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m512i;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            [[nodiscard]]
            constexpr i32x16() noexcept requires (flag.has_flag(itAVX512F)) {}

            [[nodiscard]]
            constexpr i32x16(const reg_type& m) noexcept requires (flag.has_flag(itAVX512F)) : m{ m } {}

            [[nodiscard]]
            constexpr i32x16(const i32x4& m0, const i32x4& m1, const i32x4& m2, const i32x4& m3) noexcept requires (flag.has_flag(itAVX512F)) :
            #ifndef SIMDEN_EMULATE_INTRINSICS
                m{[&]{
                    m512i ret = mmcast<m512i>(m0.m);
                    ret = _mm512_inserti32x4(ret, m1.m, 1);
                    ret = _mm512_inserti32x4(ret, m2.m, 2);
                    ret = _mm512_inserti32x4(ret, m3.m, 3);
                    return ret;
                }()}
            #else
                //m{std::make_from_tuple<reg_type>(std::tuple_cat(m0.m,m1.m,m2.m,m3.m))}
                m{
                    m0.m[0],
                    m0.m[1],
                    m0.m[2],
                    m0.m[3],
                    m1.m[0],
                    m1.m[1],
                    m1.m[2],
                    m1.m[3],
                    m2.m[0],
                    m2.m[1],
                    m2.m[2],
                    m2.m[3],
                    m3.m[0],
                    m3.m[1],
                    m3.m[2],
                    m3.m[3],
                }
            #endif
            {}
            
            template<std::convertible_to<elm_type>... T> requires(sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr i32x16(T... t) noexcept requires(flag.has_flag(itAVX)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{.m512i_i32{t...}}
                #else
                m{t...}
                #endif
            {}
            
            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            [[nodiscard]]
            friend inline i32x16 operator+(const i32x16& x, const i32x16& y) noexcept {
                return add(x, y);
            }

            [[nodiscard]]
            friend inline i32x16 operator*(const i32x16& x, const i32x16& y) noexcept {
                return mul(x, y);
            }

            [[nodiscard]]
            friend inline i32x16 operator&(const i32x16& x, const i32x16& y) noexcept {
                return band(x, y);
            }

            [[nodiscard]]
            friend inline i32x16 operator|(const i32x16& x, const i32x16& y) noexcept {
                return bor(x, y);
            }

            template<bool Aligned = false>
            [[nodiscard]]
            inline void store(std::array<int, elm_count>& x) const noexcept {
                #ifndef SIMDEN_EMULATE_INTRINSICS
                if constexpr (Aligned) {
                    _mm256_store_si256(reinterpret_cast<m256i*>(x.data()), m);
                }
                else {
                    _mm256_storeu_si256(reinterpret_cast<m256i*>(x.data()), m);
                }
                #else
                x = m;
                #endif
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<i32x16>);

        struct alignas(m512f) f32x16 {
            using elm_type = float;
            static constexpr size_t elm_size = 4;
            static constexpr size_t elm_count = 16;
            #ifndef SIMDEN_EMULATE_INTRINSICS
            using reg_type = m512f;
            #else
            using reg_type = std::array<elm_type, elm_count>;
            #endif

            constexpr f32x16() noexcept requires (flag.has_flag(itAVX)) {}

            [[nodiscard]]
            constexpr f32x16(const reg_type& m) noexcept requires (flag.has_flag(itAVX)) : m{ m } {}

            [[nodiscard]]
            constexpr f32x16(const f32x4& m0, const f32x4& m1, const f32x4& m2, const f32x4& m3) noexcept requires (flag.has_flag(itAVX)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{[&]{
                    m512f ret = mmcast<m512f>(m0.m);
                    ret = _mm512_insertf32x4(ret, m1.m, 1);
                    ret = _mm512_insertf32x4(ret, m2.m, 2);
                    ret = _mm512_insertf32x4(ret, m3.m, 3);
                    return ret;
                }()}
                #else
                //m{std::make_from_tuple<reg_type>(std::tuple_cat(m0.m,m1.m,m2.m,m3.m))}
                m{
                    m0.m[0],
                    m0.m[1],
                    m0.m[2],
                    m0.m[3],
                    m1.m[0],
                    m1.m[1],
                    m1.m[2],
                    m1.m[3],
                    m2.m[0],
                    m2.m[1],
                    m2.m[2],
                    m2.m[3],
                    m3.m[0],
                    m3.m[1],
                    m3.m[2],
                    m3.m[3],
                }
                #endif
            {}

            template<std::convertible_to<elm_type>... T> requires (sizeof...(T) == elm_count)
            [[nodiscard]]
            constexpr f32x16(T... t) noexcept requires (flag.has_flag(itAVX)) :
                #ifndef SIMDEN_EMULATE_INTRINSICS
                m{.m512_f32{t...}}
                #else
                m{t...}
                #endif
            {}

            [[nodiscard]]
            constexpr operator reg_type() const noexcept { return m; }

            [[nodiscard]]
            friend inline f32x16 operator+(const f32x16& x, const f32x16& y) noexcept {
                return add(x, y);
            }

            [[nodiscard]]
            friend inline f32x16 operator-(const f32x16& x, const f32x16& y) noexcept {
                return sub(x, y);
            }

            [[nodiscard]]
            friend inline f32x16 operator*(const f32x16& x, const f32x16& y) noexcept {
                return mul(x, y);
            }

            [[nodiscard]]
            friend inline f32x16 operator>=(const f32x16& x, const f32x16& y) noexcept {
                return cmp_ge(x, y);
            }
        private:
            friend intrinsics;
            reg_type m;
        };
        static_assert(simd_vec<f32x16>);

        template<class T> requires (simd_vec<T> || simd_reg<T>)
        [[nodiscard]]
        inline static void store64(void* dst, const T& src) noexcept requires (flag.has_flag(itSSE)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            _mm_storeu_si64(dst, mmcast<m128i>(detail::to_reg(src)));
            #else
            std::memcpy(dst, &src, 8);
            #endif
        }

        template<int F>
        [[nodiscard]]
        inline static f32x4 insert(const f32x4& x, float y) noexcept requires (flag.has_flag(itSSE41)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(x), std::bit_cast<int>(y), F));
            #else
            auto ret = x.m;
            ret[F] = y;
            return ret;
            #endif
        }

        template<std::same_as<i32x4> T>
        [[nodiscard]]
        constexpr static T set1(int x) noexcept requires (flag.has_flag(itSSE)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            //if consteval {
            if (std::is_constant_evaluated()) {
                return { x,x,x,x };
            }
            else {
                // SSE
                return _mm_set1_epi32(x);
            }
            #else
            return detail::emulate::set1<i32x4>(x);
            #endif
        }

        template<std::same_as<f32x4> T>
        [[nodiscard]]
        constexpr static T set1(float x) noexcept requires (flag.has_flag(itSSE)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            //if consteval {
            if (std::is_constant_evaluated()) {
                return { x,x,x,x };
            }
            else {
                return _mm_set1_ps(x);
            }
            #else
            return detail::emulate::set1<f32x4>(x);
            #endif
        }

        template<std::same_as<i32x8> T>
        [[nodiscard]]
        constexpr static T set1(int x) noexcept requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            //if consteval {
            if (std::is_constant_evaluated()) {
                return { x,x,x,x,x,x,x,x };
            }
            else {
                return _mm256_set1_epi32(x);
            }
            #else
            return detail::emulate::set1<i32x8>(x);
            #endif
        }

        template<std::same_as<f32x8> T>
        [[nodiscard]]
        constexpr static T set1(float x) noexcept requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            //if consteval {
            if (std::is_constant_evaluated()) {
                return { x,x,x,x,x,x,x,x };
            }
            else {
                return _mm256_set1_ps(x);
            }
            #else
            return detail::emulate::set1<f32x8>(x);
            #endif
        }

        template<std::same_as<f32x16> T>
        [[nodiscard]]
        constexpr static T set1(float x, mask16 m) noexcept requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            //if consteval {
            if (std::is_constant_evaluated()) {
                typename f32x16::reg_type ret;
                for (size_t i = 0; i < f32x16::elm_count; i++) {
                    if (m[i]) {
                        ret[i] = x;
                    }
                    else {
                        ret[i] = 0.f;
                    }
                }
            }
            else {
                return mmcast<m512f>(_mm512_maskz_set1_epi32(m, std::bit_cast<int>(x)));
            }
            #else
            typename f32x16::reg_type ret;
            for (size_t i = 0; i < f32x16::elm_count; i++) {
                if (m[i]) {
                    ret[i] = x;
                }
                else {
                    ret[i] = 0.f;
                }
            }
            #endif
        }

        template<std::same_as<f32x16> T>
        [[nodiscard]]
        constexpr static T set1(const f32x16& s, float x, const mask16& m) noexcept requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return mmcast<m512f>(_mm512_mask_set1_epi32(mmcast<m512i>(detail::to_reg(s)), m, std::bit_cast<int>(x)));
            #else
            typename f32x16::reg_type ret;
            const auto& sm = detail::to_reg(s);
            for (size_t i = 0; i < f32x16::elm_count; i++) {
                if (m[i]) {
                    ret[i] = x;
                }
                else {
                    ret[i] = sm[i];
                }
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        constexpr static std::same_as<i32x16> auto set1(int x) noexcept requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            //if consteval {
            if (std::is_constant_evaluated()) {
                return { x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x };
            }
            else {
                return _mm512_set1_epi32(x);
            }
            #else
            return detail::emulate::set1<i32x16>(x);
            #endif
        }

        template<size_t F>
        [[nodiscard]]
        inline static f32x8 insert(const f32x8& x, float y) requires (flag.has_flag(itAVX)) {
            auto low = _mm256_castps256_ps128(x);
            auto high = _mm256_extractf128_ps(x, 1);
            const int yi = std::bit_cast<int>(y);
            if constexpr (F < 4) {
                low = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(low), yi, F));
            }
            else {
                high = _mm_castsi128_ps(_mm_insert_epi32(_mm_castps_si128(high), yi, F - 4));
            }
            return _mm256_set_m128(high, low);
        }

        template<size_t F> requires (flag.has_flag(itAVX))
        [[nodiscard]]
        inline static f32x8 insert(const f32x8& x, const f32x4& y) {
            return _mm256_insertf128_ps(x, y, F);
        }

        template<std::same_as<int> T, size_t F> requires (flag.has_flag(itAVX))
        [[nodiscard]]
        inline static T extract(const i32x8& x) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            if constexpr (F < 4) {
                const auto m1 = _mm256_castsi256_si128(x);
                return _mm_extract_epi32(m1, F);
            }
            else {
                const auto m1 = _mm256_extracti128_si256(x, 1);
                return _mm_extract_epi32(m1, F - 4);
            }
            #else
            return x.m[F];
            #endif
        }

        template<std::same_as<int> T, size_t F> requires (flag.has_flag(itAVX512F))
        [[nodiscard]]
        inline static T extract(const i32x16& x) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            if constexpr (F < 4) {
                const auto m1 = _mm512_castsi512_si128(x);
                return _mm_extract_epi32(m1, F);
            }
            else {
                const auto m1 = _mm512_extracti32x4_epi32(x, F / 4);
                return _mm_extract_epi32(m1, F % 4);
            }
            #else
            return x.m[F];
            #endif
        }

        template<std::same_as<i32x4> T, size_t F> requires (flag.has_flag(itAVX512F))
        [[nodiscard]]
        inline static T extract(const i32x16& x) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            if constexpr (F == 0) {
                return _mm512_castsi512_si128(x);
            }
            else {
                return _mm512_extracti32x4_epi32(x, F);
            }
            #else
            return i32x4{
                x.m[F * 4 + 0],
                x.m[F * 4 + 1],
                x.m[F * 4 + 2],
                x.m[F * 4 + 3],
            };
            #endif
        }


        template<std::same_as<f32x4> T, size_t F>
        [[nodiscard]]
        inline static T extract(const f32x8& x) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return mmcast<m128f>(_mm256_extractf128_si256(mmcast<m256i>(detail::to_reg(x)), F));
            #else
            return typename f32x4::reg_type{
                x.m[F*4+0],
                x.m[F*4+1],
                x.m[F*4+2],
                x.m[F*4+3],
            };
            #endif
        }

        template<size_t... F> requires (sizeof...(F) == 4)
        [[nodiscard]]
        inline static f32x8 permute(const f32x8& x) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_permute_ps(x, make_flag(2, F...));
            #else
            typename f32x8::reg_type ret;
            constexpr size_t f[] = {F...};
            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 4; j++) {
                    ret[i*4+j] = x.m[i*4 + f[j]];
                }
            }
            return ret;
            #endif
        }

        template<size_t... F> requires (sizeof...(F) == 4)
        [[nodiscard]]
        inline static f32x16 permute(const f32x16& x) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_permute_ps(x, make_flag(2, F...));
            #else
            typename f32x16::reg_type ret;
            constexpr size_t f[] = {F...};
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    ret[i*4+j] = x.m[i*4 + f[j]];
                }
            }
            return ret;
            #endif
        }

        template<size_t... F> requires (sizeof...(F) == 4)
        [[nodiscard]]
        inline static i32x8 permute(const i32x8& x) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_shuffle_epi32(x, make_flag(2, F...));
            #else
            typename i32x8::reg_type ret;
            constexpr size_t f[] = {F...};
            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 4; j++) {
                    ret[i*4+j] = x.m[i*4 + f[j]];
                }
            }
            return ret;
            #endif
        }

        template<size_t... F> requires (sizeof...(F) == 4)
        [[nodiscard]]
        inline static i32x16 permute(const i32x16& x) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_shuffle_epi32(x, make_flag(2, F...));
            #else
            typename i32x16::reg_type ret;
            constexpr size_t f[] = {F...};
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    ret[i*4+j] = x.m[i*4 + f[j]];
                }
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x8 permutev(const f32x8& x, const i32x8& m) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_permutevar_ps(x, m);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 4; j++) {
                    const auto idx = m.m[i*4+j] & 0b11;
                    ret[i*4+j] = x.m[i*4 + idx];
                }
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x16 permutev(const f32x16& x, const i32x16& m) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_permutevar_ps(x, m);
            #else
            typename f32x16::reg_type ret;
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    const auto idx = m.m[i*4+j] & 0b11;
                    ret[i*4+j] = x.m[i*4 + idx];
                }
            }
            return ret;
            #endif
        }

        
        [[nodiscard]]
        inline static i32x16 permutex(const i32x16& x, const i32x16& f) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_permutexvar_epi32(f, x);
            #else
            typename i32x16::reg_type ret;
            for (size_t i = 0; i < i32x16::elm_count; i++) {
                ret[i] = x.m[f.m[i] & 0b1111];
            }
            return ret;
            #endif
        }

        template<size_t... F> requires (sizeof...(F) == 8)
        [[nodiscard]]
        static f32x8 blend(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_blend_ps(x, y, make_flag(1, F...));
        }

        template<size_t... F> requires (sizeof...(F) == 16)
        [[nodiscard]]
        static i32x16 compress(const i32x16& x) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_maskz_compress_epi32(mask16(F...), x);
            #else
            constexpr size_t f[] = {F...};
            size_t count = 0;
            typename i32x16::reg_type ret;
            for (size_t i = 0; i < i32x16::elm_count; i++) {
                if (f[i] & 1) {
                    ret[count] = x.m[i];
                    count++;
                }
            }
            for (size_t i = count; i < i32x16::elm_count; i++) {
                ret[i] = 0;
            }
            return ret;
            #endif
        }

        #ifndef SIMDEN_EMULATE_INTRINSICS
        
        [[nodiscard]]
        inline static i32x4 add(const i32x4& x, const i32x4& y) requires (flag.has_flag(itSSE)) {
            return _mm_add_epi32(x, y);
        }

        [[nodiscard]]
        inline static i32x8 add(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_add_epi32(x, y);
        }

        [[nodiscard]]
        inline static i32x16 add(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            return _mm512_add_epi32(x, y);
        }

        [[nodiscard]]
        inline static f32x8 add(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_add_ps(x, y);
        }

        [[nodiscard]]
        inline static f32x16 add(const f32x16& x, const f32x16& y) requires (flag.has_flag(itAVX512F)) {
            return _mm512_add_ps(x, y);
        }

        [[nodiscard]]
        inline static f32x8 sub(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_sub_ps(x, y);
        }

        [[nodiscard]]
        inline static f32x16 sub(const f32x16& x, const f32x16& y) requires (flag.has_flag(itAVX512F)) {
            return _mm512_sub_ps(x, y);
        }

        [[nodiscard]]
        inline static f32x4 mul(const f32x4& a, const f32x4& b) requires (flag.has_flag(itSSE)) {
            return _mm_mul_ps(a, b);
        }

        [[nodiscard]]
        inline static i32x8 mul(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_mullo_epi32(x, y);
        }

        [[nodiscard]]
        inline static i32x16 mul(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            return _mm512_mullo_epi32(x, y);
        }

        [[nodiscard]]
        inline static f32x8 mul(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_mul_ps(x, y);
        }

        [[nodiscard]]
        inline static f32x16 mul(const f32x16& x, const f32x16& y) requires (flag.has_flag(itAVX)) {
            return _mm512_mul_ps(x, y);
        }


        [[nodiscard]]
        inline static i32x8 band(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_and_si256(x, y);
        }

        [[nodiscard]]
        inline static f32x8 band(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_and_ps(x, y);
        }

        [[nodiscard]]
        inline static i32x16 band(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            return _mm512_and_epi32(x, y);
        }

        [[nodiscard]]
        inline static f32x8 bor(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_or_ps(x, y);
        }

        [[nodiscard]]
        inline static i32x8 bor(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_or_si256(x, y);
        }

        [[nodiscard]]
        inline static i32x16 bor(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            return _mm512_or_epi32(x, y);
        }

        #else

        template<simd_vec T>
        [[nodiscard]]
        static T add(const T& x, const T& y) {
            return detail::emulate::simd_2op(x, y, std::plus{});
        }

        template<simd_vec T>
        [[nodiscard]]
        static T sub(const T& x, const T& y) {
            return detail::emulate::simd_2op(x, y, std::minus{});
        }

        template<simd_vec T>
        [[nodiscard]]
        static T mul(const T& x, const T& y) {
            return detail::emulate::simd_2op(x, y, std::multiplies{});
        }

        template<simd_vec T>
        [[nodiscard]]
        static T band(const T& x, const T& y) {
            return detail::emulate::simd_2op(x, y, std::bit_and{});
        }

        template<simd_vec T>
        [[nodiscard]]
        static T bor(const T& x, const T& y) {
            return detail::emulate::simd_2op(x, y, std::bit_or{});
        }

        #endif


        #ifndef SIMDEN_EMULATE_INTRINSICS
        template<int F>
        [[nodiscard]]
        inline static f32x8 cmp(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return _mm256_cmp_ps(x, y, F);
        }

        template<int F>
        [[nodiscard]]
        inline static mask16 cmp(const f32x16& x, const f32x16& y) requires (flag.has_flag(itAVX512F)) {
            return _mm512_cmp_ps_mask(x, y, F);
        }
        #endif

        [[nodiscard]]
        inline static f32x8 cmp_ge(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return cmp<_CMP_GE_OQ>(x, y);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < f32x8::elm_count; i++) {
                ret[i] = std::bit_cast<float>(x.m[i] >= y.m[i] ? 0xffffffffu : 0u);
            }
            return ret;
            #endif
        }

        inline static mask16 cmp_ge(const f32x16& x, const f32x16& y) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return cmp<_CMP_GE_OQ>(x, y);
            #else
            typename mask16::reg_type ret;
            for (size_t i = 0; i < mask16::elm_count; i++) {
                ret[i] = x.m[i] >= y.m[i];
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x8 fma(const f32x8& x, const f32x8& y, const f32x8& z) requires (flag.has_flag(itFMA)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_fmadd_ps(x, y, z);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < f32x8::elm_count; i++) {
                ret[i] = std::fma(x.m[i], y.m[i], z.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x16 fma(const f32x16& x, const f32x16& y, const f32x16& z) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_fmadd_ps(x, y, z);
            #else
            typename f32x16::reg_type ret;
            for (size_t i = 0; i < f32x16::elm_count; i++) {
                ret[i] = std::fma(x.m[i], y.m[i], z.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x8 floor(const f32x8& x) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_floor_ps(x);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < f32x8::elm_count; i++) {
                ret[i] = std::floor(x.m[i]);
            }
            return ret;
            #endif
        }
        
        [[nodiscard]]
        inline static f32x16 floor(const f32x16& x) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_floor_ps(x);
            #else
            typename f32x16::reg_type ret;
            for (size_t i = 0; i < f32x16::elm_count; i++) {
                ret[i] = std::floor(x.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x8 round(const f32x8& x) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_round_ps(x, _MM_FROUND_NEARBYINT);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < f32x8::elm_count; i++) {
                ret[i] = std::round(x.m[i]);
            }
            return ret;
            #endif
        }
        
        [[nodiscard]]
        inline static f32x16 round(const f32x16& x) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_nearbyint_ps(x);
            #else
            typename f32x16::reg_type ret;
            for (size_t i = 0; i < f32x16::elm_count; i++) {
                ret[i] = std::round(x.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static i32x8 (min)(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_min_epi32(x, y);
            #else
            typename i32x8::reg_type ret;
            for (size_t i = 0; i < i32x8::elm_count; i++) {
                ret[i] = (std::min)(x.m[i], y.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static i32x16 (min)(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_min_epi32(x, y);
            #else
            typename i32x16::reg_type ret;
            for (size_t i = 0; i < i32x16::elm_count; i++) {
                ret[i] = (std::min)(x.m[i], y.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static i32x8 min_(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            return (min)(x, y);
        }

        [[nodiscard]]
        inline static i32x16 min_(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            return (min)(x, y);
        }

        [[nodiscard]]
        inline static f32x8 (min)(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_min_ps(x, y);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < f32x8::elm_count; i++) {
                ret[i] = (std::min)(x.m[i], y.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x8 min_(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return (min)(x, y);
        }

        [[nodiscard]]
        inline static i32x8 (max)(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_max_epi32(x, y);
            #else
            typename i32x8::reg_type ret;
            for (size_t i = 0; i < i32x8::elm_count; i++) {
                ret[i] = (std::max)(x.m[i], y.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static i32x16 (max)(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm512_max_epi32(x, y);
            #else
            typename i32x16::reg_type ret;
            for (size_t i = 0; i < i32x16::elm_count; i++) {
                ret[i] = (std::max)(x.m[i], y.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static i32x8 max_(const i32x8& x, const i32x8& y) requires (flag.has_flag(itAVX)) {
            return (max)(x, y);
        }

        [[nodiscard]]
        inline static i32x16 max_(const i32x16& x, const i32x16& y) requires (flag.has_flag(itAVX512F)) {
            return (max)(x, y);
        }
        
        [[nodiscard]]
        inline static f32x8 (max)(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_max_ps(x, y);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < 8; i++) {
                ret[i] = (std::max)(x.m[i], y.m[i]);
            }
            return ret;
            #endif
        }

        [[nodiscard]]
        inline static f32x8 max_(const f32x8& x, const f32x8& y) requires (flag.has_flag(itAVX)) {
            return (max)(x, y);
        }

        [[nodiscard]]
        inline static i32x8 clamp(const i32x8& x, const i32x8& mn, const i32x8& mx) requires (flag.has_flag(itAVX)) {
            return min_(max_(x, mn), mx);
        }

        [[nodiscard]]
        inline static i32x16 clamp(const i32x16& x, const i32x16& mn, const i32x16& mx) requires (flag.has_flag(itAVX512F)) {
            return min_(max_(x, mn), mx);
        }

        [[nodiscard]]
        inline static f32x8 clamp(const f32x8& x, const f32x8& mn, const f32x8& mx) requires (flag.has_flag(itAVX)) {
            return min_(max_(x, mn), mx);
        }

        template<size_t Scale = 4>
        [[nodiscard]]
        static i32x8 gather(const int* ptr, const i32x8& ofs) requires (flag.has_flag(itAVX2)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_i32gather_epi32(ptr, ofs, Scale);
            #else
            typename i32x8::reg_type ret;
            for (size_t i = 0; i < 8; i++) {
                ret[i] = *reinterpret_cast<const int*>(reinterpret_cast<uintptr_t>(ptr) + ofs.m[i] * Scale);
            }
            return ret;
            #endif
        }

        template<size_t Scale = 4>
        [[nodiscard]]
        static f32x8 gather(const float* ptr, const i32x8& ofs) requires (flag.has_flag(itAVX2)) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return _mm256_i32gather_ps(ptr, ofs, Scale);
            #else
            typename f32x8::reg_type ret;
            for (size_t i = 0; i < 8; i++) {
                ret[i] = *reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(ptr) + ofs.m[i] * Scale);
            }
            return ret;
            #endif
        }

        #ifndef SIMDEN_EMULATE_INTRINSICS
        template<std::same_as<i32x4> T>
        [[nodiscard]]
        static T convert_to(const u8x4& x) requires (flag.has_flag(itSSE41)) {
            return _mm_cvtepu8_epi32(x);
        }

        template<std::same_as<i32x8> T>
        [[nodiscard]]
        static T convert_to(const u8x8& x) requires (flag.has_flag(itAVX2)) {
            return _mm256_cvtepu8_epi32(x);
        }

        template<std::same_as<i32x16> T>
        [[nodiscard]]
        static T convert_to(const u8x16& x) requires (flag.has_flag(itAVX512F)) {
            return _mm512_cvtepu8_epi32(x);
        }

        template<std::same_as<f32x8> T>
        [[nodiscard]]
        static T convert_to(const i32x8& x) requires (flag.has_flag(itAVX)) {
            return _mm256_cvtepi32_ps(x);
        }
        
        template<std::same_as<f32x16> T>
        [[nodiscard]]
        static T convert_to(const i32x16& x) requires (flag.has_flag(itAVX512F)) {
            return _mm512_cvtepi32_ps(x);
        }

        template<std::same_as<i32x8> T>
        [[nodiscard]]
        static T convert_to(const f32x8& x) requires (flag.has_flag(itAVX)) {
            return _mm256_cvtps_epi32(x);
        }

        template<std::same_as<i32x16> T>
        [[nodiscard]]
        static T convert_to(const f32x16& x) requires (flag.has_flag(itAVX512F)) {
            return _mm512_cvtps_epi32(x);
        }

        template<std::same_as<f32x8> T>
        [[nodiscard]]
        static T convert_to(const u8x8& x) {
            return convert_to<f32x8>(convert_to<i32x8>(x));
        }

        template<std::same_as<f32x16> T>
        [[nodiscard]]
        static T convert_to(const u8x16& x) {
            return convert_to<f32x16>(convert_to<i32x16>(x));
        }

        template<std::same_as<u8x8> T>
        [[nodiscard]]
        static T convert_to(const i32x8& x) {
            static constexpr m256i mask_u32_to_u8{ .m256i_i8{
                0, 4, 8, 12, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
                -1,-1,-1,-1, 0, 4, 8, 12, -1,-1,-1,-1, -1,-1,-1,-1,
            } };
            const auto m1 = _mm256_shuffle_epi8(x, mask_u32_to_u8);
            const auto m2 = _mm256_extracti128_si256(m1, 0);
            const auto m3 = _mm256_extracti128_si256(m1, 1);
            const auto m4 = _mm_blend_epi32(m2, m3, make_flag(1, 0, 1, 0, 0));
            return m4;
        }

        template<std::same_as<u8x16> T>
        [[nodiscard]]
        static T convert_to(const i32x16& x) requires (flag.has_flag(itAVX512F)) {
            return _mm512_cvtepi32_epi8(x);
        }

        template<std::same_as<i32x16> T>
        [[nodiscard]]
        static T convert_to(const mask16& x) noexcept requires(flag.has_flag(itAVX512F) && flag.has_flag(itAVX512DQ)) {
            return _mm512_movm_epi32(x);
        }

        #else
        template<simd_vec To, simd_vec From> requires (To::elm_count == From::elm_count)
        [[nodiscard]]
        static To convert_to(const From& x) {
            typename To::reg_type ret;
            const auto& xm = detail::to_reg(x);
            for (size_t i = 0; i < To::elm_count; i++) {
                ret[i] = static_cast<To::elm_type>(xm[i]);
            }
            return ret;
        }

        #endif

        template<simd_vec To, simd_vec From>
        [[nodiscard]]
        static To cast_to(const From& from) {
            #ifndef SIMDEN_EMULATE_INTRINSICS
            return mmcast<To::reg_type>(static_cast<From::reg_type>(from));
            #else
            constexpr auto to_size = To::elm_size * To::elm_count;
            constexpr auto from_size = From::elm_size * From::elm_count;
            const From::reg_type& from_reg = from;
            typename To::reg_type ret;

            if constexpr (to_size == from_size) {
                std::memcpy(&ret, &from, to_size);
            }
            else if constexpr (to_size < from_size) {
                std::memcpy(&ret, &from, to_size);
            }
            else {
                std::memset(&ret, 0, to_size);
                std::memcpy(&ret, &from, from_size);
            }

            return ret;
            #endif
        }
    };
}
