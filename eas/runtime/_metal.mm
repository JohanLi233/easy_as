#define PY_SSIZE_T_CLEAN
#include <Python.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CoreFoundation/CoreFoundation.h>

namespace {

static const char* kPipelineCapsuleName = "eas._metal.pipeline";
static const char* kPendingCapsuleName = "eas._metal.pending";

struct Pipeline {
  void* device;
  void* queue;
  void* pso;
};

struct Pending {
  void* cb;
  void* buffers;
  Py_buffer* views;
  Py_ssize_t argc;
};

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
  p->argc = 0;
  delete p;
}

static PyObject* is_available(PyObject* /*self*/, PyObject* /*args*/) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
      Py_RETURN_TRUE;
    }
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices && devices.count > 0) {
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
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      device = devices.firstObject;
    }
    if (!device) {
      PyErr_SetString(PyExc_RuntimeError, "no Metal devices available");
      return nullptr;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
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

    auto* pending = new Pending();
    pending->cb = (__bridge_retained void*)cb;
    pending->buffers = (__bridge_retained void*)buffers;
    pending->views = views;
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
