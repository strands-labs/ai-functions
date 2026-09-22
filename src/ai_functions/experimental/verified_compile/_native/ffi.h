/* Private ABI between the Python bridge and verified native functions. */
#ifndef AI_FUNCTIONS_VERIFIED_FFI_H
#define AI_FUNCTIONS_VERIFIED_FFI_H

#include <lean/lean.h>
#include <stdint.h>

#define AV_ABI_VERSION 1u
#define AV_ABI_MAGIC UINT64_C(0x4149564552494649)

enum av_kind { AV_INT = 1, AV_BOOL = 2, AV_FLOAT = 3, AV_INT_LIST = 4 };

typedef union {
    lean_object *object;
    double floating;
    uint8_t boolean;
} av_value;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t arity;
    /* Input kinds followed by the output kind. */
    const uint8_t *signature;
    lean_object *(*initialize)(uint8_t builtin);
    /* Consumes every object argument and returns an owned object result. */
    void (*invoke)(const av_value *arguments, av_value *result);
} av_descriptor;

typedef const av_descriptor *(*av_descriptor_getter)(void);

#endif
