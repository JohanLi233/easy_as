#define PY_SSIZE_T_CLEAN
#include <Python.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CoreFoundation/CoreFoundation.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

static const char* kPipelineCapsuleName = "eas._metal.pipeline";
static const char* kPendingCapsuleName = "eas._metal.pending";
static const char* kBufferCapsuleName = "eas._metal.buffer";

struct Pipeline {
  void* device;
  void* queue;
  void* pso;
};

struct DLManagedTensor;

struct Buffer {
  void* buf;
  Py_ssize_t nbytes;
  Py_ssize_t offset;
  int storage;  // 0=shared, 1=private
  DLManagedTensor* dlpack;  // holds an owned DLManagedTensor (optional)
};

// Minimal DLPack structs (no external headers).
// https://dmlc.github.io/dlpack/latest/
static constexpr int kDLMetal = 8;
static constexpr uint8_t kDLFloat = 2;

struct DLDevice {
  int device_type;
  int device_id;
};

struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor {
  void* data;
  DLDevice device;
  int ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};

struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensor* self);
};

struct Pending {
  void* cb;
  void* buffers;
  Py_buffer* views;
  PyObject** keepalive;
  Py_ssize_t argc;
};

static void buffer_capsule_destructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, kBufferCapsuleName);
  if (!ptr) {
    return;
  }
  auto* b = static_cast<Buffer*>(ptr);
  if (b->dlpack) {
    if (b->dlpack->deleter) {
      b->dlpack->deleter(b->dlpack);
    }
    b->dlpack = nullptr;
    b->buf = nullptr;
    delete b;
    return;
  }
  @autoreleasepool {
    if (b->buf) {
      CFRelease(b->buf);
      b->buf = nullptr;
    }
  }
  delete b;
}

static void pipeline_capsule_destructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, kPipelineCapsuleName);
  if (!ptr) {
    return;
  }
  auto* p = static_cast<Pipeline*>(ptr);
  if (p->pso) {
    CFRelease(p->pso);
  }
  if (p->queue) {
    CFRelease(p->queue);
  }
  if (p->device) {
    CFRelease(p->device);
  }
  delete p;
}

static void pending_capsule_destructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, kPendingCapsuleName);
  if (!ptr) {
    return;
  }
  auto* p = static_cast<Pending*>(ptr);
  @autoreleasepool {
    if (p->cb) {
      id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)p->cb;
      [cb waitUntilCompleted];
      CFRelease(p->cb);
      p->cb = nullptr;
    }
    if (p->buffers) {
      CFRelease(p->buffers);
      p->buffers = nullptr;
    }
  }

  if (p->views) {
    for (Py_ssize_t i = 0; i < p->argc; i++) {
      if (p->views[i].obj) {
        PyBuffer_Release(&p->views[i]);
      }
    }
    free(p->views);
    p->views = nullptr;
  }
  if (p->keepalive) {
    for (Py_ssize_t i = 0; i < p->argc; i++) {
      Py_XDECREF(p->keepalive[i]);
    }
    free(p->keepalive);
    p->keepalive = nullptr;
  }
  p->argc = 0;
  delete p;
}

static id<MTLDevice> get_device() {
  static id<MTLDevice> device = nil;
  static std::once_flag once;
  std::call_once(once, []() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      if (!device) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        device = devices.firstObject;
      }
    }
  });
  return device;
}

static id<MTLCommandQueue> get_queue() {
  static id<MTLCommandQueue> queue = nil;
  static std::once_flag once;
  std::call_once(once, []() {
    @autoreleasepool {
      id<MTLDevice> device = get_device();
      if (device) {
        queue = [device newCommandQueue];
      }
    }
  });
  return queue;
}

static PyObject* is_available(PyObject* /*self*/, PyObject* /*args*/) {
  @autoreleasepool {
    if (get_device()) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
  }
}

static PyObject* compile(PyObject* /*self*/, PyObject* args) {
  const char* msl_src_c = nullptr;
  const char* kernel_name_c = nullptr;
  if (!PyArg_ParseTuple(args, "ss", &msl_src_c, &kernel_name_c)) {
    return nullptr;
  }

  @autoreleasepool {
    id<MTLDevice> device = get_device();
    if (!device) {
      PyErr_SetString(PyExc_RuntimeError, "no Metal devices available");
      return nullptr;
    }
    id<MTLCommandQueue> queue = get_queue();
    if (!queue) {
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLCommandQueue");
      return nullptr;
    }

    NSString* msl_src = [NSString stringWithUTF8String:msl_src_c];
    NSString* kernel_name = [NSString stringWithUTF8String:kernel_name_c];

    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:msl_src options:opts error:&err];
    if (!library) {
      const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
      PyErr_Format(PyExc_RuntimeError, "Metal compile failed: %s", msg);
      return nullptr;
    }

    id<MTLFunction> fn = [library newFunctionWithName:kernel_name];
    if (!fn) {
      PyErr_Format(PyExc_RuntimeError, "kernel function not found: %s", kernel_name_c);
      return nullptr;
    }

    err = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
      PyErr_Format(PyExc_RuntimeError, "failed to create compute pipeline: %s", msg);
      return nullptr;
    }

    auto* pipeline = new Pipeline();
    pipeline->device = (__bridge_retained void*)device;
    pipeline->queue = (__bridge_retained void*)queue;
    pipeline->pso = (__bridge_retained void*)pso;

    return PyCapsule_New(pipeline, kPipelineCapsuleName, pipeline_capsule_destructor);
  }
}

static PyObject* alloc_buffer(PyObject* /*self*/, PyObject* args) {
  long long nbytes_ll = 0;
  const char* storage_c = nullptr;
  if (!PyArg_ParseTuple(args, "Ls", &nbytes_ll, &storage_c)) {
    return nullptr;
  }
  if (nbytes_ll <= 0) {
    PyErr_SetString(PyExc_ValueError, "nbytes must be > 0");
    return nullptr;
  }
  @autoreleasepool {
    id<MTLDevice> device = get_device();
    if (!device) {
      PyErr_SetString(PyExc_RuntimeError, "no Metal devices available");
      return nullptr;
    }
    NSString* storage = [NSString stringWithUTF8String:storage_c ? storage_c : ""];
    MTLResourceOptions opts = 0;
    int storage_tag = 0;
    if ([storage isEqualToString:@"private"]) {
      opts = MTLResourceStorageModePrivate;
      storage_tag = 1;
    } else if ([storage isEqualToString:@"shared"]) {
      opts = MTLResourceStorageModeShared;
      storage_tag = 0;
    } else {
      PyErr_Format(PyExc_ValueError, "unsupported storage: %s (expected 'private'|'shared')", storage_c);
      return nullptr;
    }
    id<MTLBuffer> buf = [device newBufferWithLength:(NSUInteger)nbytes_ll options:opts];
    if (!buf) {
      PyErr_SetString(PyExc_RuntimeError, "failed to allocate MTLBuffer");
      return nullptr;
    }
    auto* b = new Buffer();
    b->buf = (__bridge_retained void*)buf;
    b->nbytes = (Py_ssize_t)nbytes_ll;
    b->offset = 0;
    b->storage = storage_tag;
    b->dlpack = nullptr;
    return PyCapsule_New(b, kBufferCapsuleName, buffer_capsule_destructor);
  }
}

static bool copy_from_host_impl(Buffer* b, Py_buffer* src_view) {
  if (src_view->len <= 0 || src_view->buf == nullptr) {
    PyErr_SetString(PyExc_ValueError, "source buffer cannot be empty");
    return false;
  }
  if (src_view->len > b->nbytes) {
    PyErr_SetString(PyExc_ValueError, "source is larger than destination buffer");
    return false;
  }
  @autoreleasepool {
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)b->buf;
    if (b->storage == 0) {
      char* base = static_cast<char*>(dst.contents);
      if (!base) {
        PyErr_SetString(PyExc_RuntimeError, "MTLBuffer contents is null");
        return false;
      }
      memcpy(base + b->offset, src_view->buf, (size_t)src_view->len);
      return true;
    }
    id<MTLDevice> device = get_device();
    id<MTLCommandQueue> queue = get_queue();
    if (!device || !queue) {
      PyErr_SetString(PyExc_RuntimeError, "Metal device/queue unavailable");
      return false;
    }
    id<MTLBuffer> staging = [device newBufferWithBytes:src_view->buf
                                                length:(NSUInteger)src_view->len
                                               options:MTLResourceStorageModeShared];
    if (!staging) {
      PyErr_SetString(PyExc_RuntimeError, "failed to allocate staging buffer");
      return false;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) {
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLCommandBuffer");
      return false;
    }
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) {
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLBlitCommandEncoder");
      return false;
    }
    [blit copyFromBuffer:staging
            sourceOffset:0
                toBuffer:dst
       destinationOffset:(NSUInteger)b->offset
                    size:(NSUInteger)src_view->len];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
      NSError* err = cb.error;
      const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
      PyErr_Format(PyExc_RuntimeError, "Metal blit failed: %s", msg);
      return false;
    }
    return true;
  }
}

static bool copy_to_host_impl(Buffer* b, Py_buffer* dst_view) {
  if (dst_view->len <= 0 || dst_view->buf == nullptr) {
    PyErr_SetString(PyExc_ValueError, "destination buffer cannot be empty");
    return false;
  }
  if (dst_view->len < b->nbytes) {
    PyErr_SetString(PyExc_ValueError, "destination is smaller than source buffer");
    return false;
  }
  @autoreleasepool {
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)b->buf;
    if (b->storage == 0) {
      char* base = static_cast<char*>(src.contents);
      if (!base) {
        PyErr_SetString(PyExc_RuntimeError, "MTLBuffer contents is null");
        return false;
      }
      memcpy(dst_view->buf, base + b->offset, (size_t)b->nbytes);
      return true;
    }
    id<MTLDevice> device = get_device();
    id<MTLCommandQueue> queue = get_queue();
    if (!device || !queue) {
      PyErr_SetString(PyExc_RuntimeError, "Metal device/queue unavailable");
      return false;
    }
    id<MTLBuffer> staging = [device newBufferWithLength:(NSUInteger)b->nbytes options:MTLResourceStorageModeShared];
    if (!staging) {
      PyErr_SetString(PyExc_RuntimeError, "failed to allocate staging buffer");
      return false;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) {
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLCommandBuffer");
      return false;
    }
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) {
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLBlitCommandEncoder");
      return false;
    }
    [blit copyFromBuffer:src
            sourceOffset:(NSUInteger)b->offset
                toBuffer:staging
       destinationOffset:0
                    size:(NSUInteger)b->nbytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
      NSError* err = cb.error;
      const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
      PyErr_Format(PyExc_RuntimeError, "Metal blit failed: %s", msg);
      return false;
    }
    memcpy(dst_view->buf, staging.contents, (size_t)b->nbytes);
    return true;
  }
}

static PyObject* copy_from_host(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* src_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &src_obj)) {
    return nullptr;
  }
  void* ptr = PyCapsule_GetPointer(capsule, kBufferCapsuleName);
  if (!ptr) {
    return nullptr;
  }
  auto* b = static_cast<Buffer*>(ptr);
  Py_buffer view;
  if (PyObject_GetBuffer(src_obj, &view, PyBUF_CONTIG_RO) != 0) {
    return nullptr;
  }
  bool ok = copy_from_host_impl(b, &view);
  PyBuffer_Release(&view);
  if (!ok) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyObject* copy_to_host(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* dst_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &dst_obj)) {
    return nullptr;
  }
  void* ptr = PyCapsule_GetPointer(capsule, kBufferCapsuleName);
  if (!ptr) {
    return nullptr;
  }
  auto* b = static_cast<Buffer*>(ptr);
  Py_buffer view;
  if (PyObject_GetBuffer(dst_obj, &view, PyBUF_CONTIG | PyBUF_WRITABLE) != 0) {
    return nullptr;
  }
  bool ok = copy_to_host_impl(b, &view);
  PyBuffer_Release(&view);
  if (!ok) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

static bool dlpack_is_f32(const DLDataType& dt) {
  return dt.code == kDLFloat && dt.bits == 32 && dt.lanes == 1;
}

static bool dlpack_is_c_contig(const DLTensor& t) {
  if (!t.shape || t.ndim <= 0) {
    return false;
  }
  if (!t.strides) {
    return true;
  }
  int64_t expected = 1;
  for (int i = t.ndim - 1; i >= 0; i--) {
    const int64_t dim = t.shape[i];
    if (dim <= 0) {
      return false;
    }
    if (t.strides[i] != expected) {
      return false;
    }
    expected *= dim;
  }
  return true;
}

static bool dlpack_numel(const DLTensor& t, uint64_t* out) {
  if (!t.shape || t.ndim <= 0) {
    return false;
  }
  uint64_t numel = 1;
  for (int i = 0; i < t.ndim; i++) {
    const int64_t dim = t.shape[i];
    if (dim <= 0) {
      return false;
    }
    const uint64_t udim = static_cast<uint64_t>(dim);
    if (udim != 0 && numel > (UINT64_MAX / udim)) {
      return false;
    }
    numel *= udim;
  }
  *out = numel;
  return true;
}

static PyObject* dlpack_import(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  if (!PyCapsule_CheckExact(capsule)) {
    PyErr_SetString(PyExc_TypeError, "expected a DLPack capsule (PyCapsule)");
    return nullptr;
  }

  const char* cap_name = PyCapsule_GetName(capsule);
  if (!cap_name || strcmp(cap_name, "dltensor") != 0) {
    PyErr_SetString(PyExc_ValueError, "expected a DLPack capsule with name 'dltensor'");
    return nullptr;
  }

  void* ptr = PyCapsule_GetPointer(capsule, "dltensor");
  if (!ptr) {
    return nullptr;
  }
  auto* mt = static_cast<DLManagedTensor*>(ptr);
  const DLTensor& t = mt->dl_tensor;

  if (t.device.device_type != kDLMetal) {
    PyErr_Format(
        PyExc_ValueError, "dlpack_import expects kDLMetal device_type=%d", kDLMetal);
    return nullptr;
  }
  if (!dlpack_is_f32(t.dtype)) {
    PyErr_SetString(PyExc_TypeError, "dlpack_import only supports float32 tensors");
    return nullptr;
  }
  if (!dlpack_is_c_contig(t)) {
    PyErr_SetString(PyExc_ValueError, "dlpack_import requires a contiguous tensor");
    return nullptr;
  }

  uint64_t numel = 0;
  if (!dlpack_numel(t, &numel)) {
    PyErr_SetString(PyExc_ValueError, "invalid dlpack shape");
    return nullptr;
  }
  const uint64_t offset = t.byte_offset;
  if ((offset % 4) != 0) {
    PyErr_SetString(PyExc_ValueError, "dlpack byte_offset must be multiple of 4");
    return nullptr;
  }
  if (numel > (UINT64_MAX / 4)) {
    PyErr_SetString(PyExc_OverflowError, "dlpack tensor too large");
    return nullptr;
  }
  const uint64_t nbytes = numel * 4;

  @autoreleasepool {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)t.data;
    if (!buf) {
      PyErr_SetString(PyExc_ValueError, "dlpack tensor has null data pointer");
      return nullptr;
    }
    const uint64_t buf_len = static_cast<uint64_t>(buf.length);
    if (offset > buf_len || (buf_len - offset) < nbytes) {
      PyErr_SetString(PyExc_ValueError, "dlpack view exceeds MTLBuffer length");
      return nullptr;
    }

    int storage_tag = 1;
    if (buf.storageMode == MTLStorageModeShared) {
      storage_tag = 0;
    } else {
      storage_tag = 1;
    }

    auto* b = new Buffer();
    b->buf = t.data;
    b->nbytes = (Py_ssize_t)nbytes;
    b->offset = (Py_ssize_t)offset;
    b->storage = storage_tag;
    b->dlpack = mt;

    if (PyCapsule_SetName(capsule, "used_dltensor") != 0) {
      delete b;
      return nullptr;
    }
    if (PyCapsule_SetDestructor(capsule, nullptr) != 0) {
      delete b;
      return nullptr;
    }

    PyObject* buf_capsule = PyCapsule_New(b, kBufferCapsuleName, buffer_capsule_destructor);
    if (!buf_capsule) {
      return nullptr;
    }
    PyObject* shape_tuple = PyTuple_New(t.ndim);
    if (!shape_tuple) {
      Py_DECREF(buf_capsule);
      return nullptr;
    }
    for (int i = 0; i < t.ndim; i++) {
      PyObject* dim = PyLong_FromLongLong((long long)t.shape[i]);
      if (!dim) {
        Py_DECREF(shape_tuple);
        Py_DECREF(buf_capsule);
        return nullptr;
      }
      PyTuple_SET_ITEM(shape_tuple, i, dim);
    }

    PyObject* out = PyTuple_New(2);
    if (!out) {
      Py_DECREF(shape_tuple);
      Py_DECREF(buf_capsule);
      return nullptr;
    }
    PyTuple_SET_ITEM(out, 0, buf_capsule);
    PyTuple_SET_ITEM(out, 1, shape_tuple);
    return out;
  }
}

struct DlpackExportCtx {
  PyObject* buf_capsule;
  int64_t* shape;
  int ndim;
};

static void dlpack_export_deleter(DLManagedTensor* self) {
  if (!self) {
    return;
  }
  auto* ctx = static_cast<DlpackExportCtx*>(self->manager_ctx);
  if (ctx) {
    if (ctx->shape) {
      free(ctx->shape);
      ctx->shape = nullptr;
    }
    if (ctx->buf_capsule) {
      PyGILState_STATE st = PyGILState_Ensure();
      Py_DECREF(ctx->buf_capsule);
      PyGILState_Release(st);
      ctx->buf_capsule = nullptr;
    }
    delete ctx;
  }
  delete self;
}

static void dlpack_capsule_destructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, "dltensor");
  if (!ptr) {
    PyErr_Clear();
    return;
  }
  auto* mt = static_cast<DLManagedTensor*>(ptr);
  if (mt->deleter) {
    mt->deleter(mt);
  }
}

static bool queue_synchronize_impl() {
  @autoreleasepool {
    id<MTLCommandQueue> queue = get_queue();
    if (!queue) {
      PyErr_SetString(PyExc_RuntimeError, "Metal command queue unavailable");
      return false;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) {
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLCommandBuffer");
      return false;
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
      NSError* err = cb.error;
      const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
      PyErr_Format(PyExc_RuntimeError, "Metal command buffer failed: %s", msg);
      return false;
    }
    return true;
  }
}

static PyObject* queue_synchronize(PyObject* /*self*/, PyObject* /*args*/) {
  if (!queue_synchronize_impl()) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyObject* dlpack_export(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* shape_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &shape_obj)) {
    return nullptr;
  }
  void* ptr = PyCapsule_GetPointer(capsule, kBufferCapsuleName);
  if (!ptr) {
    return nullptr;
  }
  auto* b = static_cast<Buffer*>(ptr);

  PyObject* shape = PySequence_Fast(shape_obj, "shape must be a sequence of ints");
  if (!shape) {
    return nullptr;
  }
  const Py_ssize_t ndim = PySequence_Fast_GET_SIZE(shape);
  if (ndim <= 0) {
    Py_DECREF(shape);
    PyErr_SetString(PyExc_ValueError, "shape must be non-empty");
    return nullptr;
  }

  uint64_t numel = 1;
  int64_t* shape_arr = static_cast<int64_t*>(calloc((size_t)ndim, sizeof(int64_t)));
  if (!shape_arr) {
    Py_DECREF(shape);
    PyErr_NoMemory();
    return nullptr;
  }
  bool ok = true;
  for (Py_ssize_t i = 0; i < ndim; i++) {
    PyObject* item = PySequence_Fast_GET_ITEM(shape, i);
    long long dim_ll = PyLong_AsLongLong(item);
    if (dim_ll <= 0 || PyErr_Occurred()) {
      ok = false;
      break;
    }
    shape_arr[i] = (int64_t)dim_ll;
    const uint64_t udim = static_cast<uint64_t>(dim_ll);
    if (udim != 0 && numel > (UINT64_MAX / udim)) {
      ok = false;
      PyErr_SetString(PyExc_OverflowError, "tensor too large");
      break;
    }
    numel *= udim;
  }
  Py_DECREF(shape);
  if (!ok) {
    free(shape_arr);
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "invalid shape");
    }
    return nullptr;
  }
  if (numel > (UINT64_MAX / 4)) {
    free(shape_arr);
    PyErr_SetString(PyExc_OverflowError, "tensor too large");
    return nullptr;
  }
  const uint64_t nbytes = numel * 4;
  if (nbytes > static_cast<uint64_t>(b->nbytes)) {
    free(shape_arr);
    PyErr_SetString(PyExc_ValueError, "shape exceeds buffer view size");
    return nullptr;
  }

  if (!queue_synchronize_impl()) {
    free(shape_arr);
    return nullptr;
  }

  auto* mt = new DLManagedTensor();
  memset(mt, 0, sizeof(*mt));
  mt->dl_tensor.data = b->buf;
  mt->dl_tensor.device.device_type = kDLMetal;
  mt->dl_tensor.device.device_id = 0;
  mt->dl_tensor.ndim = (int)ndim;
  mt->dl_tensor.dtype.code = kDLFloat;
  mt->dl_tensor.dtype.bits = 32;
  mt->dl_tensor.dtype.lanes = 1;
  mt->dl_tensor.shape = shape_arr;
  mt->dl_tensor.strides = nullptr;
  mt->dl_tensor.byte_offset = (uint64_t)b->offset;

  auto* ctx = new DlpackExportCtx();
  ctx->shape = shape_arr;
  ctx->ndim = (int)ndim;
  Py_INCREF(capsule);
  ctx->buf_capsule = capsule;
  mt->manager_ctx = ctx;
  mt->deleter = dlpack_export_deleter;

  PyObject* out = PyCapsule_New(mt, "dltensor", dlpack_capsule_destructor);
  if (!out) {
    // ctx->shape is owned by mt and will be freed by deleter if we call it.
    mt->deleter(mt);
    return nullptr;
  }
  return out;
}

static PyObject* launch_async(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* argv_obj = nullptr;
  PyObject* writable_obj = nullptr;
  int grid = 0;
  int threads = 0;
  if (!PyArg_ParseTuple(args, "OOOii", &capsule, &argv_obj, &writable_obj, &grid, &threads)) {
    return nullptr;
  }
  if (grid <= 0 || threads <= 0) {
    PyErr_SetString(PyExc_ValueError, "grid and threads must be > 0");
    return nullptr;
  }

  void* ptr = PyCapsule_GetPointer(capsule, kPipelineCapsuleName);
  if (!ptr) {
    return nullptr;
  }
  auto* pipeline = static_cast<Pipeline*>(ptr);

  PyObject* argv = PySequence_Fast(argv_obj, "argv must be a sequence");
  if (!argv) {
    return nullptr;
  }
  PyObject* writable = PySequence_Fast(writable_obj, "writable must be a sequence");
  if (!writable) {
    Py_DECREF(argv);
    return nullptr;
  }

  Py_ssize_t argc = PySequence_Fast_GET_SIZE(argv);
  if (PySequence_Fast_GET_SIZE(writable) != argc) {
    Py_DECREF(writable);
    Py_DECREF(argv);
    PyErr_SetString(PyExc_ValueError, "writable must have same length as argv");
    return nullptr;
  }

  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)pipeline->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)pipeline->queue;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline->pso;

    if (threads > (int)pso.maxTotalThreadsPerThreadgroup) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_Format(
          PyExc_ValueError,
          "threads (%d) exceeds maxTotalThreadsPerThreadgroup (%lu)",
          threads,
          (unsigned long)pso.maxTotalThreadsPerThreadgroup);
      return nullptr;
    }

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLCommandBuffer");
      return nullptr;
    }

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLComputeCommandEncoder");
      return nullptr;
    }
    [enc setComputePipelineState:pso];

    NSMutableArray<id<MTLBuffer>>* buffers = [NSMutableArray arrayWithCapacity:(NSUInteger)argc];
    Py_buffer* views = static_cast<Py_buffer*>(calloc((size_t)argc, sizeof(Py_buffer)));
    PyObject** keepalive = static_cast<PyObject**>(calloc((size_t)argc, sizeof(PyObject*)));
    if (!views) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_NoMemory();
      return nullptr;
    }
    if (!keepalive) {
      free(views);
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_NoMemory();
      return nullptr;
    }

    bool ok = true;
    for (Py_ssize_t i = 0; i < argc; i++) {
      PyObject* obj = PySequence_Fast_GET_ITEM(argv, i);
      PyObject* w = PySequence_Fast_GET_ITEM(writable, i);
      int want_write = PyObject_IsTrue(w);
      if (want_write < 0) {
        ok = false;
        break;
      }

      // Scalars are passed from Python as small bytes objects. Binding them via
      // -setBytes:length:atIndex: avoids creating an MTLBuffer per launch.
      if (!want_write && (PyBytes_Check(obj) || PyByteArray_Check(obj))) {
        char* data = nullptr;
        Py_ssize_t len = 0;
        if (PyBytes_Check(obj)) {
          if (PyBytes_AsStringAndSize(obj, &data, &len) != 0) {
            ok = false;
            break;
          }
        } else {
          len = PyByteArray_Size(obj);
          data = PyByteArray_AsString(obj);
          if (len < 0 || data == nullptr) {
            PyErr_SetString(PyExc_ValueError, "invalid bytearray argument");
            ok = false;
            break;
          }
        }
        if (len <= 0) {
          PyErr_SetString(PyExc_ValueError, "scalar argument cannot be empty");
          ok = false;
          break;
        }
        [enc setBytes:data length:(NSUInteger)len atIndex:(NSUInteger)i];
        continue;
      }

      // Reusable device buffers are passed as capsules.
      if (PyCapsule_CheckExact(obj)) {
        void* bptr = PyCapsule_GetPointer(obj, kBufferCapsuleName);
        if (bptr) {
          auto* b = static_cast<Buffer*>(bptr);
          id<MTLBuffer> buf = (__bridge id<MTLBuffer>)b->buf;
          if (!buf) {
            PyErr_SetString(PyExc_RuntimeError, "invalid Metal buffer capsule");
            ok = false;
            break;
          }
          [enc setBuffer:buf offset:(NSUInteger)b->offset atIndex:(NSUInteger)i];
          Py_INCREF(obj);
          keepalive[i] = obj;
          continue;
        }
        if (PyErr_Occurred()) {
          ok = false;
          break;
        }
      }

      int flags = want_write ? PyBUF_CONTIG : PyBUF_CONTIG_RO;
      if (PyObject_GetBuffer(obj, &views[i], flags) != 0) {
        ok = false;
        break;
      }
      if (views[i].len <= 0 || views[i].buf == nullptr) {
        PyErr_SetString(PyExc_ValueError, "buffer argument cannot be empty");
        ok = false;
        break;
      }

      id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:views[i].buf
                                                    length:(NSUInteger)views[i].len
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
      if (!buf) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create MTLBuffer");
        ok = false;
        break;
      }
      [buffers addObject:buf];
      [enc setBuffer:buf offset:0 atIndex:(NSUInteger)i];
    }

    if (!ok) {
      for (Py_ssize_t i = 0; i < argc; i++) {
        if (views[i].obj) {
          PyBuffer_Release(&views[i]);
        }
        Py_XDECREF(keepalive[i]);
      }
      free(keepalive);
      free(views);
      Py_DECREF(writable);
      Py_DECREF(argv);
      return nullptr;
    }

    MTLSize threadgroups = MTLSizeMake((NSUInteger)grid, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake((NSUInteger)threads, 1, 1);
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [enc endEncoding];

    [cb commit];

    auto* pending = new Pending();
    pending->cb = (__bridge_retained void*)cb;
    pending->buffers = (__bridge_retained void*)buffers;
    pending->views = views;
    pending->keepalive = keepalive;
    pending->argc = argc;

    Py_DECREF(writable);
    Py_DECREF(argv);
    return PyCapsule_New(pending, kPendingCapsuleName, pending_capsule_destructor);
  }
}

static PyObject* synchronize(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  void* ptr = PyCapsule_GetPointer(capsule, kPendingCapsuleName);
  if (!ptr) {
    return nullptr;
  }
  auto* p = static_cast<Pending*>(ptr);
  if (!p->cb) {
    Py_RETURN_NONE;
  }

  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)p->cb;
    [cb waitUntilCompleted];
    bool ok = (cb.status != MTLCommandBufferStatusError);
    NSError* err = cb.error;

    if (p->buffers) {
      CFRelease(p->buffers);
      p->buffers = nullptr;
    }
    if (p->cb) {
      CFRelease(p->cb);
      p->cb = nullptr;
    }

    if (p->views) {
      for (Py_ssize_t i = 0; i < p->argc; i++) {
        PyBuffer_Release(&p->views[i]);
      }
      free(p->views);
      p->views = nullptr;
    }
    if (p->keepalive) {
      for (Py_ssize_t i = 0; i < p->argc; i++) {
        Py_XDECREF(p->keepalive[i]);
      }
      free(p->keepalive);
      p->keepalive = nullptr;
    }
    p->argc = 0;

    if (!ok) {
      const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
      PyErr_Format(PyExc_RuntimeError, "Metal command buffer failed: %s", msg);
      return nullptr;
    }
    Py_RETURN_NONE;
  }
}

static PyObject* launch(PyObject* /*self*/, PyObject* args) {
  PyObject* capsule = nullptr;
  PyObject* argv_obj = nullptr;
  PyObject* writable_obj = nullptr;
  int grid = 0;
  int threads = 0;
  if (!PyArg_ParseTuple(args, "OOOii", &capsule, &argv_obj, &writable_obj, &grid, &threads)) {
    return nullptr;
  }
  if (grid <= 0 || threads <= 0) {
    PyErr_SetString(PyExc_ValueError, "grid and threads must be > 0");
    return nullptr;
  }

  void* ptr = PyCapsule_GetPointer(capsule, kPipelineCapsuleName);
  if (!ptr) {
    return nullptr;
  }
  auto* pipeline = static_cast<Pipeline*>(ptr);

  PyObject* argv = PySequence_Fast(argv_obj, "argv must be a sequence");
  if (!argv) {
    return nullptr;
  }
  PyObject* writable = PySequence_Fast(writable_obj, "writable must be a sequence");
  if (!writable) {
    Py_DECREF(argv);
    return nullptr;
  }

  Py_ssize_t argc = PySequence_Fast_GET_SIZE(argv);
  if (PySequence_Fast_GET_SIZE(writable) != argc) {
    Py_DECREF(writable);
    Py_DECREF(argv);
    PyErr_SetString(PyExc_ValueError, "writable must have same length as argv");
    return nullptr;
  }

  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)pipeline->device;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)pipeline->queue;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline->pso;

    if (threads > (int)pso.maxTotalThreadsPerThreadgroup) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_Format(
          PyExc_ValueError,
          "threads (%d) exceeds maxTotalThreadsPerThreadgroup (%lu)",
          threads,
          (unsigned long)pso.maxTotalThreadsPerThreadgroup);
      return nullptr;
    }

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLCommandBuffer");
      return nullptr;
    }

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_SetString(PyExc_RuntimeError, "failed to create MTLComputeCommandEncoder");
      return nullptr;
    }
    [enc setComputePipelineState:pso];

    NSMutableArray<id<MTLBuffer>>* buffers = [NSMutableArray arrayWithCapacity:(NSUInteger)argc];
    Py_buffer* views = static_cast<Py_buffer*>(calloc((size_t)argc, sizeof(Py_buffer)));
    if (!views) {
      Py_DECREF(writable);
      Py_DECREF(argv);
      PyErr_NoMemory();
      return nullptr;
    }

    bool ok = true;
    for (Py_ssize_t i = 0; i < argc; i++) {
      PyObject* obj = PySequence_Fast_GET_ITEM(argv, i);
      PyObject* w = PySequence_Fast_GET_ITEM(writable, i);
      int want_write = PyObject_IsTrue(w);
      if (want_write < 0) {
        ok = false;
        break;
      }

      // Scalars are passed from Python as small bytes objects. Binding them via
      // -setBytes:length:atIndex: avoids creating an MTLBuffer per launch.
      if (!want_write && (PyBytes_Check(obj) || PyByteArray_Check(obj))) {
        char* data = nullptr;
        Py_ssize_t len = 0;
        if (PyBytes_Check(obj)) {
          if (PyBytes_AsStringAndSize(obj, &data, &len) != 0) {
            ok = false;
            break;
          }
        } else {
          len = PyByteArray_Size(obj);
          data = PyByteArray_AsString(obj);
          if (len < 0 || data == nullptr) {
            PyErr_SetString(PyExc_ValueError, "invalid bytearray argument");
            ok = false;
            break;
          }
        }
        if (len <= 0) {
          PyErr_SetString(PyExc_ValueError, "scalar argument cannot be empty");
          ok = false;
          break;
        }
        [enc setBytes:data length:(NSUInteger)len atIndex:(NSUInteger)i];
        continue;
      }

      // Reusable device buffers are passed as capsules.
      if (PyCapsule_CheckExact(obj)) {
        void* bptr = PyCapsule_GetPointer(obj, kBufferCapsuleName);
        if (bptr) {
          auto* b = static_cast<Buffer*>(bptr);
          id<MTLBuffer> buf = (__bridge id<MTLBuffer>)b->buf;
          if (!buf) {
            PyErr_SetString(PyExc_RuntimeError, "invalid Metal buffer capsule");
            ok = false;
            break;
          }
          [enc setBuffer:buf offset:(NSUInteger)b->offset atIndex:(NSUInteger)i];
          continue;
        }
        if (PyErr_Occurred()) {
          ok = false;
          break;
        }
      }

      int flags = want_write ? PyBUF_CONTIG : PyBUF_CONTIG_RO;
      if (PyObject_GetBuffer(obj, &views[i], flags) != 0) {
        ok = false;
        break;
      }
      if (views[i].len <= 0 || views[i].buf == nullptr) {
        PyErr_SetString(PyExc_ValueError, "buffer argument cannot be empty");
        ok = false;
        break;
      }

      id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:views[i].buf
                                                    length:(NSUInteger)views[i].len
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
      if (!buf) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create MTLBuffer");
        ok = false;
        break;
      }
      [buffers addObject:buf];
      [enc setBuffer:buf offset:0 atIndex:(NSUInteger)i];
    }

    if (!ok) {
      for (Py_ssize_t i = 0; i < argc; i++) {
        if (views[i].obj) {
          PyBuffer_Release(&views[i]);
        }
      }
      free(views);
      Py_DECREF(writable);
      Py_DECREF(argv);
      return nullptr;
    }

    MTLSize threadgroups = MTLSizeMake((NSUInteger)grid, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake((NSUInteger)threads, 1, 1);
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
      NSError* err = cb.error;
      const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
      PyErr_Format(PyExc_RuntimeError, "Metal command buffer failed: %s", msg);
      for (Py_ssize_t i = 0; i < argc; i++) {
        if (views[i].obj) {
          PyBuffer_Release(&views[i]);
        }
      }
      free(views);
      Py_DECREF(writable);
      Py_DECREF(argv);
      return nullptr;
    }

    for (Py_ssize_t i = 0; i < argc; i++) {
      PyBuffer_Release(&views[i]);
    }
    free(views);
    Py_DECREF(writable);
    Py_DECREF(argv);
    Py_RETURN_NONE;
  }
}

static PyMethodDef Methods[] = {
    {"is_available", is_available, METH_NOARGS, nullptr},
    {"compile", compile, METH_VARARGS, nullptr},
    {"alloc_buffer", alloc_buffer, METH_VARARGS, nullptr},
    {"copy_from_host", copy_from_host, METH_VARARGS, nullptr},
    {"copy_to_host", copy_to_host, METH_VARARGS, nullptr},
    {"dlpack_import", dlpack_import, METH_VARARGS, nullptr},
    {"dlpack_export", dlpack_export, METH_VARARGS, nullptr},
    {"queue_synchronize", queue_synchronize, METH_NOARGS, nullptr},
    {"launch", launch, METH_VARARGS, nullptr},
    {"launch_async", launch_async, METH_VARARGS, nullptr},
    {"synchronize", synchronize, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_metal",
    nullptr,
    -1,
    Methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__metal() { return PyModule_Create(&Module); }
