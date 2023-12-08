#pragma once

#include <cassert>
#include <stdexcept>

// Macros used for defining symbol visibility, only GLIBC is supported
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define CURADON_EXPORT __attribute__((visibility("default")))
#define CURADON_HIDDEN __attribute__((visibility("hidden")))
#else
#define CURADON_EXPORT
#define CURADON_HIDDEN
#endif

namespace curad {
struct logic_error : public std::logic_error {
    using std::logic_error::logic_error;
};

struct cuda_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class bad_alloc : public std::bad_alloc {
  public:
    bad_alloc(const char *msg) : what_{std::string{std::bad_alloc::what()} + ": " + msg} {}
    bad_alloc(std::string const &msg) : bad_alloc{msg.c_str()} {}

    [[nodiscard]]
    const char *what() const noexcept override {
        return what_.c_str();
    }

  private:
    std::string what_;
};

class out_of_memory : public bad_alloc {
  public:
    out_of_memory(const char *msg) : bad_alloc{std::string{"out_of_memory: "} + msg} {}
    out_of_memory(std::string const &msg) : out_of_memory{msg.c_str()} {}
};

class out_of_range : public std::out_of_range {
    using std::out_of_range::out_of_range;
};

} // namespace curad

#define STRINGIFY_DETAIL(x) #x
#define CURADON_STRINGIFY(x) STRINGIFY_DETAIL(x)

#define CURADON_EXPECTS(...)                                                                       \
    GET_CURADON_EXPECTS_MACRO(__VA_ARGS__, CURADON_EXPECTS_3, CURADON_EXPECTS_2)                   \
    (__VA_ARGS__)
#define GET_CURADON_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME
#define CURADON_EXPECTS_3(_condition, _exception_type, _reason)                                    \
    (!!(_condition)) ? static_cast<void>(0)                                                        \
                     : throw _exception_type /*NOLINT(bugprone-macro-parentheses)*/                \
    {                                                                                              \
        "RMM failure at: " __FILE__ ":" CURADON_STRINGIFY(__LINE__) ": " _reason                   \
    }
#define CURADON_EXPECTS_2(_condition, _reason)                                                     \
    CURADON_EXPECTS_3(_condition, curad::logic_error, _reason)

#define CURADON_FAIL(...)                                                                          \
    GET_CURADON_FAIL_MACRO(__VA_ARGS__, CURADON_FAIL_2, CURADON_FAIL_1)                            \
    (__VA_ARGS__)
#define GET_CURADON_FAIL_MACRO(_1, _2, NAME, ...) NAME
#define CURADON_FAIL_2(_what, _exception_type)                                                     \
    /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                                 \
    throw _exception_type{"RMM failure at:" __FILE__ ":" CURADON_STRINGIFY(__LINE__) ": " _what};
#define CURADON_FAIL_1(_what) CURADON_FAIL_2(_what, curad::logic_error)

#define CURADON_CUDA_TRY(...)                                                                      \
    GET_CURADON_CUDA_TRY_MACRO(__VA_ARGS__, CURADON_CUDA_TRY_2, CURADON_CUDA_TRY_1)                \
    (__VA_ARGS__)
#define GET_CURADON_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CURADON_CUDA_TRY_2(_call, _exception_type)                                                 \
    do {                                                                                           \
        cudaError_t const error = (_call);                                                         \
        if (cudaSuccess != error) {                                                                \
            cudaGetLastError();                                                                    \
            throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                \
                                  CURADON_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) +   \
                                  " " + cudaGetErrorString(error)};                                \
        }                                                                                          \
    } while (0)
#define CURADON_CUDA_TRY_1(_call) CURADON_CUDA_TRY_2(_call, curad::cuda_error)

#define CURADON_CUDA_TRY_ALLOC(_call)                                                              \
    do {                                                                                           \
        cudaError_t const error = (_call);                                                         \
        if (cudaSuccess != error) {                                                                \
            cudaGetLastError();                                                                    \
            auto const msg = std::string{"CUDA error at: "} + __FILE__ + ":" +                     \
                             CURADON_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " +  \
                             cudaGetErrorString(error);                                            \
            if (cudaErrorMemoryAllocation == error) {                                              \
                throw curad::out_of_memory{msg};                                                   \
            } else {                                                                               \
                throw curad::bad_alloc{msg};                                                       \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#ifdef NDEBUG
#define CURADON_ASSERT_CUDA_SUCCESS(_call)                                                         \
    do {                                                                                           \
        (_call);                                                                                   \
    } while (0);
#else
#define CURADON_ASSERT_CUDA_SUCCESS(_call)                                                         \
    do {                                                                                           \
        cudaError_t const status__ = (_call);                                                      \
        if (status__ != cudaSuccess) {                                                             \
            std::cerr << "CUDA Error detected. " << cudaGetErrorName(status__) << " "              \
                      << cudaGetErrorString(status__) << std::endl;                                \
        }                                                                                          \
        assert(status__ == cudaSuccess);                                                           \
    } while (0)
#endif

#ifdef NDEBUG
#define CURADON_LOGGING_ASSERT(_expr) (void)0
#elif SPDLOG_ACTIVE_LEVEL < SPDLOG_LEVEL_OFF
#define CURADON_LOGGING_ASSERT(_expr)                                                              \
    do {                                                                                           \
        bool const success = (_expr);                                                              \
        if (!success) {                                                                            \
            CURADON_LOG_CRITICAL("[" __FILE__ ":" CURADON_STRINGIFY(                               \
                __LINE__) "] Assertion " CURADON_STRINGIFY(_expr) " failed.");                     \
            curad::logger().flush();                                                               \
            assert(success);                                                                       \
        }                                                                                          \
    } while (0)
#else
#define CURADON_LOGGING_ASSERT(_expr) assert((_expr));
#endif
