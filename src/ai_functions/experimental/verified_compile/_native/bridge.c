/*
 * Direct CPython/Lean bridge. No Python or Lean source is evaluated here.
 *
 * Built for each supported CPython minor version, using that interpreter's
 * integer layout. The Lean/GMP layout is pinned to Lean 4.33.1 and checked at
 * initialization. Large integers are repacked directly between their digit
 * arrays into the destination storage, without a temporary value buffer.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cpython/longintrepr.h>
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include "ffi.h"

#if PY_VERSION_HEX < 0x030C0000
#error "The verified-function bridge requires CPython 3.12 or newer"
#endif
#ifdef Py_GIL_DISABLED
#error "The verified-function bridge requires the GIL"
#endif

/* Lean's pinned mpz_object is a lean_object followed by GMP's mpz_t.
 * Use the GMP allocator exported by the SAME Lean shared library: linking a
 * second GMP copy could mix allocators. No additional GMP library is needed.
 * Layout sources: Lean src/runtime/{object.h,mpz.h}, GMP's public mpz_t ABI.
 */
typedef struct {
    int allocated;
    int size;
    uint64_t *limbs;
} av_mpz;
extern void *__gmpz_realloc(av_mpz *, ptrdiff_t);
extern void __gmpz_init(av_mpz *);
extern void __gmpz_clear(av_mpz *);
extern lean_object *lean_alloc_mpz(av_mpz *);
extern void lean_initialize_runtime_module(void);
extern void lean_initialize_thread(void);
extern void lean_finalize_thread(void);

_Static_assert(sizeof(void *) == 8, "Only 64-bit runtimes are supported");
_Static_assert(sizeof(unsigned long) == 8, "The pinned GMP uses 64-bit limbs");
_Static_assert(sizeof(lean_object) % _Alignof(av_mpz) == 0, "GMP field alignment");

static pid_t runtime_pid;
static int64_t interpreter_id;
static int runtime_ready;
static pthread_key_t thread_key;

typedef struct {
    const av_descriptor *descriptor;
    PyObject *keys;
    void *library;
    pid_t pid;
} av_function;

static av_mpz *mpz_value(lean_object *value) {
    return (av_mpz *)((char *)value + sizeof(lean_object));
}

static unsigned bit_length(uint64_t value) {
    return value ? 64u - (unsigned)__builtin_clzll(value) : 0u;
}

static double normalize_float(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000)
        && (bits & UINT64_C(0x000fffffffffffff))) {
        bits = UINT64_C(0x7ff8000000000000);
        memcpy(&value, &bits, sizeof(value));
    }
    return value;
}

static void finalize_thread(void *initialized) {
    if (initialized && getpid() == runtime_pid) lean_finalize_thread();
}

static int ensure_thread(void) {
    if (getpid() != runtime_pid) {
        PyErr_SetString(PyExc_RuntimeError, "Native functions require a fresh process after fork; use spawn");
        return -1;
    }
    if (PyInterpreterState_GetID(PyThreadState_GetInterpreter(PyThreadState_Get())) != interpreter_id) {
        PyErr_SetString(PyExc_RuntimeError, "Native functions cannot be shared between Python interpreters");
        return -1;
    }
    if (!pthread_getspecific(thread_key)) {
        lean_initialize_thread();
        if (pthread_setspecific(thread_key, (void *)1)) {
            lean_finalize_thread();
            PyErr_SetString(PyExc_RuntimeError, "Could not initialize the native execution thread");
            return -1;
        }
    }
    return 0;
}

static lean_object *integer_from_python(PyObject *value) {
    if (!PyLong_CheckExact(value)) {
        PyErr_SetString(PyExc_TypeError, "Native integer arguments must be exact Python ints");
        return NULL;
    }
    int overflow = 0;
    long long small = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (!overflow) {
        if (small == -1 && PyErr_Occurred()) return NULL;
        if (small >= LEAN_MIN_SMALL_INT && small <= LEAN_MAX_SMALL_INT)
            return lean_int64_to_int((int64_t)small);
    }

    PyLongObject *number = (PyLongObject *)value;
    Py_ssize_t count = (Py_ssize_t)(number->long_value.lv_tag >> _PyLong_NON_SIZE_BITS);
    const digit *digits = number->long_value.ob_digit;
    size_t bits = (size_t)(count - 1) * PyLong_SHIFT + bit_length(digits[count - 1]);
    size_t limb_count = (bits + 63) / 64;
    if (limb_count > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "Integer exceeds the native runtime's allocation limit");
        return NULL;
    }
    /* Construct a real mpz object; integer constructors canonicalize zero to
     * an immediate. Copying an empty mpz allocates no temporary digit buffer. */
    av_mpz empty;
    __gmpz_init(&empty);
    lean_object *result = lean_alloc_mpz(&empty);
    __gmpz_clear(&empty);
    av_mpz *target = mpz_value(result);
    __gmpz_realloc(target, (ptrdiff_t)limb_count);
    uint64_t accumulator = 0;
    unsigned occupied = 0;
    size_t output = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        uint64_t next = digits[i];
        accumulator |= next << occupied;
        occupied += PyLong_SHIFT;
        if (occupied >= 64) {
            target->limbs[output++] = accumulator;
            occupied -= 64;
            accumulator = occupied ? next >> (PyLong_SHIFT - occupied) : 0;
        }
    }
    if (output < limb_count) target->limbs[output] = accumulator;
    target->size = (number->long_value.lv_tag & _PyLong_SIGN_MASK) == 2
        ? -(int)limb_count : (int)limb_count;
    return result;
}

static PyObject *integer_to_python(lean_object *value) {
    if (lean_is_scalar(value)) return PyLong_FromLongLong(lean_scalar_to_int64(value));
    if (!lean_is_mpz(value)) {
        PyErr_SetString(PyExc_RuntimeError, "Native function returned an invalid integer");
        return NULL;
    }
    const av_mpz *number = mpz_value(value);
    size_t count = number->size < 0 ? (size_t)(-(int64_t)number->size) : (size_t)number->size;
    if (!count) return PyLong_FromLong(0);
    size_t bits = (count - 1) * 64 + bit_length(number->limbs[count - 1]);
    size_t digit_count = (bits + PyLong_SHIFT - 1) / PyLong_SHIFT;
    if (digit_count > PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "Native integer exceeds Python's allocation limit");
        return NULL;
    }
#if PY_VERSION_HEX >= 0x030E0000
    void *storage;
    PyLongWriter *writer = PyLongWriter_Create(number->size < 0, (Py_ssize_t)digit_count, &storage);
    if (!writer) return NULL;
    digit *digits = storage;
#else
    PyLongObject *result = _PyLong_New((Py_ssize_t)digit_count);
    if (!result) return NULL;
    digit *digits = result->long_value.ob_digit;
#endif
    for (size_t i = 0; i < digit_count; ++i) {
        size_t offset = i * PyLong_SHIFT;
        size_t limb = offset / 64;
        unsigned shift = (unsigned)(offset % 64);
        uint64_t part = number->limbs[limb] >> shift;
        if (shift + PyLong_SHIFT > 64 && limb + 1 < count)
            part |= number->limbs[limb + 1] << (64 - shift);
        digits[i] = (digit)(part & PyLong_MASK);
    }
#if PY_VERSION_HEX >= 0x030E0000
    return PyLongWriter_Finish(writer);
#else
    if (number->size < 0)
        result->long_value.lv_tag = (result->long_value.lv_tag & ~(uintptr_t)_PyLong_SIGN_MASK) | 2;
    return (PyObject *)result;
#endif
}

static lean_object *list_from_python(PyObject *value) {
    if (!PyList_CheckExact(value)) {
        PyErr_SetString(PyExc_TypeError, "Native list arguments must be exact Python lists");
        return NULL;
    }
    lean_object *result = lean_box(0);
    /* Borrow Python elements while the GIL is held; allocate only Lean nodes. */
    for (Py_ssize_t i = PyList_GET_SIZE(value); i-- > 0;) {
        lean_object *item = integer_from_python(PyList_GET_ITEM(value, i));
        if (!item) { lean_dec(result); return NULL; }
        lean_object *node = lean_alloc_ctor(1, 2, 0);
        lean_ctor_set(node, 0, item);
        lean_ctor_set(node, 1, result);
        result = node;
    }
    return result;
}

static PyObject *list_to_python(lean_object *value) {
    Py_ssize_t count = 0;
    lean_object *cursor = value;
    while (!lean_is_scalar(cursor)) {
        if (lean_ptr_tag(cursor) != 1 || lean_ctor_num_objs(cursor) != 2 || count == PY_SSIZE_T_MAX) {
            PyErr_SetString(PyExc_RuntimeError, "Native function returned an invalid integer list");
            return NULL;
        }
        ++count;
        cursor = lean_ctor_get(cursor, 1);
    }
    if (lean_unbox(cursor) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Native function returned an invalid list terminator");
        return NULL;
    }
    PyObject *result = PyList_New(count);
    if (!result) return NULL;
    cursor = value;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *item = integer_to_python(lean_ctor_get(cursor, 0));
        if (!item) { Py_DECREF(result); return NULL; }
        PyList_SET_ITEM(result, i, item); /* steals the new reference */
        cursor = lean_ctor_get(cursor, 1);
    }
    return result;
}

static void release_argument(uint8_t kind, av_value value) {
    if (kind == AV_INT || kind == AV_INT_LIST) lean_dec(value.object);
}

static PyObject *invoke(PyObject *capsule, PyObject *const *python_args, Py_ssize_t count) {
    av_function *function = PyCapsule_GetPointer(capsule, "ai_functions.verified.function");
    if (!function || ensure_thread()) return NULL;
    if (count != 1 || !PyDict_CheckExact(python_args[0])) {
        PyErr_SetString(PyExc_TypeError, "Native invocation requires one bound-argument dictionary");
        return NULL;
    }
    const av_descriptor *descriptor = function->descriptor;
    if (PyDict_Size(python_args[0]) != descriptor->arity) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of native arguments");
        return NULL;
    }
    av_value stack_values[8];
    av_value *values = descriptor->arity <= 8 ? stack_values : PyMem_Calloc(descriptor->arity, sizeof(av_value));
    if (!values) return PyErr_NoMemory();
    uint32_t initialized = 0;
    for (; initialized < descriptor->arity; ++initialized) {
        PyObject *value = PyDict_GetItemWithError(python_args[0], PyTuple_GET_ITEM(function->keys, initialized));
        if (!value) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "Missing native argument");
            goto failed;
        }
        switch (descriptor->signature[initialized]) {
        case AV_INT:
            values[initialized].object = integer_from_python(value);
            if (!values[initialized].object) goto failed;
            break;
        case AV_INT_LIST:
            values[initialized].object = list_from_python(value);
            if (!values[initialized].object) goto failed;
            break;
        case AV_FLOAT:
            if (!PyFloat_CheckExact(value)) {
                PyErr_SetString(PyExc_TypeError, "Native float arguments must be exact Python floats");
                goto failed;
            }
            values[initialized].floating = normalize_float(PyFloat_AS_DOUBLE(value));
            break;
        case AV_BOOL:
            if (!PyBool_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "Native Boolean arguments must be Python bools");
                goto failed;
            }
            values[initialized].boolean = value == Py_True;
            break;
        }
    }
    av_value output;
    Py_BEGIN_ALLOW_THREADS
    descriptor->invoke(values, &output); /* ownership of object arguments moves into Lean */
    Py_END_ALLOW_THREADS
    if (values != stack_values) PyMem_Free(values);
    PyObject *result;
    switch (descriptor->signature[descriptor->arity]) {
    case AV_INT:
        result = integer_to_python(output.object);
        lean_dec(output.object);
        return result;
    case AV_INT_LIST:
        result = list_to_python(output.object);
        lean_dec(output.object);
        return result;
    case AV_FLOAT:
        return PyFloat_FromDouble(normalize_float(output.floating));
    case AV_BOOL:
        if (output.boolean > 1) {
            PyErr_SetString(PyExc_RuntimeError, "Native function returned an invalid Boolean");
            return NULL;
        }
        return PyBool_FromLong(output.boolean);
    }
    PyErr_SetString(PyExc_RuntimeError, "Invalid native result kind");
    return NULL;
failed:
    for (uint32_t i = 0; i < initialized; ++i) release_argument(descriptor->signature[i], values[i]);
    if (values != stack_values) PyMem_Free(values);
    return NULL;
}

static void destroy_function(PyObject *capsule) {
    av_function *function = PyCapsule_GetPointer(capsule, "ai_functions.verified.function");
    if (!function) { PyErr_Clear(); return; }
    Py_XDECREF(function->keys);
    /* Successfully initialized DSOs remain loaded. Lean module constants can
     * contain code pointers. Retaining a tiny DSO avoids dangling pointers;
     * no compiler Environment or .olean data is loaded or retained. */
    PyMem_Free(function);
}

static PyMethodDef invoke_method = {
    "invoke", (PyCFunction)(void (*)(void))invoke, METH_FASTCALL,
    "Invoke one verified function using typed native arguments."
};

static PyObject *load_function(PyObject *self, PyObject *args) {
    (void)self;
    const char *path;
    PyObject *signature;
    if (!PyArg_ParseTuple(args, "sO!:load", &path, &PyBytes_Type, &signature) || ensure_thread()) return NULL;
    Py_ssize_t size = PyBytes_GET_SIZE(signature);
    if (size < 1 || (size_t)size > UINT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "Invalid native function signature");
        return NULL;
    }
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) { PyErr_SetString(PyExc_ImportError, dlerror()); return NULL; }
    av_descriptor_getter getter = (av_descriptor_getter)dlsym(library, "ai_verified_descriptor_v1");
    if (!getter) {
        PyErr_SetString(PyExc_ImportError, "Verified native function has no compatible FFI descriptor");
        dlclose(library);
        return NULL;
    }
    const av_descriptor *descriptor = getter();
    if (!descriptor || descriptor->magic != AV_ABI_MAGIC || descriptor->version != AV_ABI_VERSION
        || descriptor->arity != (uint32_t)(size - 1) || !descriptor->signature
        || !descriptor->initialize || !descriptor->invoke
        || memcmp(descriptor->signature, PyBytes_AS_STRING(signature), (size_t)size)) {
        PyErr_SetString(PyExc_ImportError, "Verified native function ABI or types do not match its specification");
        dlclose(library);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        if (descriptor->signature[i] < AV_INT || descriptor->signature[i] > AV_INT_LIST) {
            PyErr_SetString(PyExc_ImportError, "Unknown verified native value type");
            dlclose(library);
            return NULL;
        }
    }
    av_function *function = PyMem_Calloc(1, sizeof(av_function));
    if (!function) { dlclose(library); return PyErr_NoMemory(); }
    function->descriptor = descriptor;
    function->library = library;
    function->pid = getpid();
    function->keys = PyTuple_New(descriptor->arity);
    if (!function->keys) { PyMem_Free(function); dlclose(library); return NULL; }
    for (uint32_t i = 0; i < descriptor->arity; ++i) {
        PyObject *key = PyUnicode_FromFormat("v%u", i);
        if (!key) { Py_DECREF(function->keys); PyMem_Free(function); dlclose(library); return NULL; }
        PyTuple_SET_ITEM(function->keys, i, key);
    }
    lean_object *initialized = descriptor->initialize(1);
    if (!lean_io_result_is_ok(initialized)) {
        lean_dec(initialized);
        Py_DECREF(function->keys);
        PyMem_Free(function);
        PyErr_SetString(PyExc_RuntimeError, "Verified native function initialization failed");
        return NULL;
    }
    lean_dec(initialized);
    lean_io_mark_end_initialization();
    PyObject *capsule = PyCapsule_New(function, "ai_functions.verified.function", destroy_function);
    if (!capsule) { Py_DECREF(function->keys); PyMem_Free(function); return NULL; }
    PyObject *result = PyCFunction_NewEx(&invoke_method, capsule, NULL);
    Py_DECREF(capsule);
    return result;
}

static PyMethodDef methods[] = {
    {"load", load_function, METH_VARARGS, "Load a previously verified native function."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_bridge",
    .m_doc = "Private typed native FFI for verified_ai_compile.",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__bridge(void) {
    if (runtime_ready < 0) {
        PyErr_SetString(PyExc_ImportError, "The native runtime failed initialization; restart Python");
        return NULL;
    }
    int64_t current_interpreter = PyInterpreterState_GetID(PyThreadState_GetInterpreter(PyThreadState_Get()));
    if (runtime_ready && (getpid() != runtime_pid || current_interpreter != interpreter_id)) {
        PyErr_SetString(PyExc_ImportError, "The native runtime cannot be reused after fork or across interpreters");
        return NULL;
    }
    if (!runtime_ready) {
        if (pthread_key_create(&thread_key, finalize_thread)) {
            PyErr_SetString(PyExc_ImportError, "Could not create native thread state");
            return NULL;
        }
        runtime_pid = getpid();
        interpreter_id = current_interpreter;
        lean_initialize_runtime_module();
        /* A failed layout/thread check must not allow a later import to skip
         * validation and reuse a partially initialized runtime. */
        runtime_ready = -1;
        if (ensure_thread()) return NULL;
        lean_object *probe = lean_big_int64_to_int(INT64_MAX);
        av_mpz *number = mpz_value(probe);
        int compatible = !lean_is_scalar(probe) && lean_is_mpz(probe) && number->size == 1
            && number->allocated >= 1 && number->limbs && number->limbs[0] == INT64_MAX;
        lean_dec(probe);
        if (!compatible) {
            PyErr_SetString(PyExc_ImportError, "Unsupported native integer layout; use the supported Lean toolchain");
            return NULL;
        }
        runtime_ready = 1;
    }
    PyObject *result = PyModule_Create(&module);
    if (result && PyModule_AddIntConstant(result, "ABI_VERSION", AV_ABI_VERSION) < 0) { Py_DECREF(result); return NULL; }
    return result;
}
