
#pragma once

#include "../include/tensorfem/tensor_fem.hpp"

/*
  Useful common functions
*/

#define pass_or_fail(check, error, tolerance)                                                                      \
  {                                                                                                                \
    auto typed_error = static_cast<typename std::remove_reference<decltype(tolerance)>::type>(error);              \
    bool this_check = boba::abs(typed_error) < tolerance;                                                          \
    std::cout << " checking " << #error << " = " << std::scientific << typed_error << " < " << tolerance << " ? "; \
    std::cout << (this_check ? ("pass") : ("fail")) << std::endl;                                                  \
    check = check && this_check;                                                                                   \
    std::cout << " cumulative check = " << (check ? ("pass") : ("fail")) << std::endl;                             \
    bool fail_immediately = boba::is_env_nonempty("FAIL_IMMEDIATELY");                                             \
    bool fail_never = boba::is_env_nonempty("FAIL_NEVER");                                                         \
    if (not(check) and fail_immediately and not(fail_never))                                                       \
    {                                                                                                              \
      boba_error("Check failed!");                                                                                 \
    }                                                                                                              \
  }

#define pass_or_fail_bool(check, condition)                                            \
  {                                                                                    \
    bool this_check = static_cast<bool>(condition);                                    \
    std::cout << " checking " << #condition << " ? "                                   \
              << (this_check ? "pass" : "fail") << std::endl;                          \
    check = check && this_check;                                                       \
    std::cout << " cumulative check = " << (check ? ("pass") : ("fail")) << std::endl; \
    bool fail_immediately = boba::is_env_nonempty("FAIL_IMMEDIATELY");                 \
    bool fail_never = boba::is_env_nonempty("FAIL_NEVER");                             \
    if (not(check) and fail_immediately and not(fail_never))                           \
    {                                                                                  \
      boba_error("Check failed!");                                                     \
    }                                                                                  \
  }

#define final_check(check) boba::is_env_nonempty("FAIL_NEVER") ? 0 : not(check);
