#define PY_SSIZE_T_CLEAN
#include <Python.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CoreFoundation/CoreFoundation.h>

namespace {

static const char* kPipelineCapsuleName = "eas._metal.pipeline";

struct Pipeline {
  void* device;
  void* queue;
  void* pso;
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

static PyObject* is_available(PyObject* /*self*/, PyObject* /*args*/) {
  @autoreleasepool {
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
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    id<MTLDevice> device = devices.firstObject;
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
