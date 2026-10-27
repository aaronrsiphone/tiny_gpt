# coding: utf-8
'''
Native 16-bit matrix-vector capability diagnostics (stage 1 of the
native-16bit investigation).

The question is whether Accelerate can multiply directly against the
memory-mapped BF16/FP16 weights used by lowmem mode, instead of widening
every slab to float32 first. Apple's headers declare the BNNS entry points
and the data-type enums but do not say which dtype combinations the fully
connected layer or BNNSMatMul accept, whether float32 activations may be
mixed with 16-bit weights, or whether BNNSFlagsUseClientPtr really avoids a
private weight copy. Only the device can answer that, so this script asks it.

Two levels, both read-only with respect to the repository:

  symbols (default)  Environment, CPU feature flags, library loading and a
                     dlsym() presence check for every candidate symbol.
                     Calls no Accelerate function at all.

  probe              Additionally calls the *documented* BNNS / vImage
                     functions (signatures transcribed from the SDK headers,
                     see docs/native16_feasibility.md) on tiny matrices
                     (at most 256 x 512) and records, per dtype combination:
                     created or refused, numeric error against NumPy,
                     bytes BNNS allocated at creation and per apply (via the
                     documented alloc_memory hook), and whether the filter
                     still reads the caller's weight buffer after creation.

No undocumented function is ever called; speculative names (e.g. a half
precision cblas) are only looked up, never invoked.

A native crash cannot be caught from Python, so before each probe step the
step name is written to a progress file. If the previous run died, the next
run reports which step it died in and skips it automatically.

Pythonista: open this file and press Run. It runs `symbols`, writes the
report, then asks whether to run `probe`. Elsewhere:

    python tools/native16_diagnostics.py [--probe] [--model DIR] [--out DIR]

Results go to native16_results/ next to the repository root (or --out):
diag_<timestamp>.json (machine readable) and diag_<timestamp>.txt.
'''

import ctypes
import datetime
import json
import os
import platform
import struct
import sys
import time
import traceback

TOOL_VERSION = 'native-16bit-diagnostic-1'

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

try:
    import numpy as np
except ImportError:          # report it rather than die before writing
    np = None

# --------------------------------------------------------------- constants
# Transcribed from bnns_constants.h (macOS 15.5 SDK; iOS 18.5 equivalent).

DT_F16 = 0x10000 | 16          # BNNSDataTypeFloat16   ios(10.0)
DT_F32 = 0x10000 | 32          # BNNSDataTypeFloat32   ios(10.0)
DT_BF16 = 0x10000 | 0x8000 | 16  # BNNSDataTypeBFloat16 ios(15.0)
DTYPE_NAMES = {DT_F16: 'F16', DT_F32: 'F32', DT_BF16: 'BF16'}
DTYPE_BYTES = {DT_F16: 2, DT_F32: 4, DT_BF16: 2}

LAYOUT_VECTOR = 0x10000           # BNNSDataLayoutVector          ios(14.0)
LAYOUT_ROW_MAJOR = 0x20000        # BNNSDataLayoutRowMajorMatrix  ios(14.0)
LAYOUT_2D_FIRST_MAJOR = 0x28001   # BNNSDataLayout2DFirstMajor    ios(14.0)

FLAG_USE_CLIENT_PTR = 0x0001      # BNNSFlagsUseClientPtr         ios(10.0)
ACT_IDENTITY = 0                  # BNNSActivationFunctionIdentity

# TINY_GPT_NATIVE16_LIB substitutes another library for Accelerate. The
# tests point it at a small fake BNNS built from tests/native16_fake_bnns.c
# so the ctypes plumbing runs on machines without Accelerate.
ACCELERATE = os.environ.get(
    'TINY_GPT_NATIVE16_LIB',
    '/System/Library/Frameworks/Accelerate.framework/Accelerate')
LIBRARIES = [
    ('Accelerate', ACCELERATE),
    ('vecLib', '/System/Library/Frameworks/Accelerate.framework/'
               'Frameworks/vecLib.framework/vecLib'),
    ('libBNNS', '/System/Library/Frameworks/Accelerate.framework/'
                'Frameworks/vecLib.framework/libBNNS.dylib'),
    ('vImage', '/System/Library/Frameworks/Accelerate.framework/'
               'Frameworks/vImage.framework/vImage'),
]

# (symbol, status) -- status says what the headers say about it.
SYMBOLS = [
    # BNNS legacy filter API: declared, deprecated since iOS 18.
    ('BNNSFilterCreateLayerFullyConnected', 'declared ios(14.0, deprecated 18.0)'),
    ('BNNSFilterApply', 'declared ios(10.0, deprecated 18.0)'),
    ('BNNSFilterApplyBatch', 'declared ios(10.0, deprecated 18.0)'),
    ('BNNSFilterDestroy', 'declared ios(10.0)'),
    ('BNNSFilterCreateFullyConnectedLayer', 'declared ios(10.0, deprecated 14.0)'),
    ('BNNSMatMul', 'declared ios(16.0, deprecated 18.0)'),
    ('BNNSMatMulWorkspaceSize', 'declared ios(16.0, deprecated 18.0)'),
    ('BNNSDirectApplyBroadcastMatMul', 'declared ios(14.0, deprecated 16.0)'),
    ('BNNSFilterCreateLayerBroadcastMatMul', 'declared ios(14.0, deprecated 18.0)'),
    ('BNNSCopy', 'declared ios(14.0)'),
    ('BNNSNDArrayGetDataSize', 'declared ios(16.0)'),
    ('BNNSTensorGetAllocationSize', 'declared ios(18.0)'),
    ('BNNSDirectApplyFullyConnectedBatch',
     'named in header comments only, not declared; never called'),
    # BNNSGraph: needs a compiled Core ML model file; presence only.
    ('BNNSGraphCompileOptionsMakeDefault', 'declared ios(18.0)'),
    ('BNNSGraphCompileFromFile', 'declared ios(18.0)'),
    ('BNNSGraphContextMake', 'declared ios(18.0)'),
    ('BNNSGraphContextExecute', 'declared ios(18.0)'),
    # BLAS used today, plus the ILP64/new-LAPACK variants.
    ('cblas_sgemv', 'declared'),
    ('cblas_sgemm', 'declared'),
    ('cblas_sgemv$NEWLAPACK', 'declared ios(16.4) new interface'),
    ('cblas_sgemv$NEWLAPACK$ILP64', 'declared ios(16.4) new interface'),
    # Speculative half/bf16 BLAS names: NOT in the 15.5 SDK headers.
    ('cblas_hgemv', 'speculative, not in SDK headers; lookup only'),
    ('cblas_hgemm', 'speculative, not in SDK headers; lookup only'),
    ('cblas_gemm_bf16bf16f32', 'speculative, not in SDK headers; lookup only'),
    # Conversion helpers.
    ('vImageConvert_Planar16FtoPlanarF', 'declared ios(5.0)'),
    ('vImageConvert_PlanarFtoPlanar16F', 'declared ios(5.0)'),
    ('vDSP_vfltu16', 'declared (integer -> float, not a bit cast)'),
]

SYSCTLS_STR = ['hw.machine', 'hw.model', 'kern.osversion',
               'kern.osproductversion', 'kern.version',
               'machdep.cpu.brand_string']
SYSCTLS_INT = ['hw.ncpu', 'hw.physicalcpu', 'hw.memsize', 'hw.pagesize',
               'hw.cachelinesize', 'hw.l1dcachesize', 'hw.l2cachesize',
               'hw.nperflevels',
               'hw.perflevel0.physicalcpu', 'hw.perflevel0.l2cachesize',
               'hw.perflevel1.physicalcpu', 'hw.perflevel1.l2cachesize',
               'hw.optional.neon', 'hw.optional.neon_fp16',
               'hw.optional.arm.FEAT_FP16', 'hw.optional.arm.FEAT_BF16',
               'hw.optional.arm.FEAT_FHM', 'hw.optional.arm.FEAT_I8MM',
               'hw.optional.arm.FEAT_DotProd', 'hw.optional.arm.FEAT_SME',
               'hw.optional.arm.FEAT_SME2', 'hw.optional.arm64',
               'hw.cpufamily', 'hw.cpusubfamily']


# ------------------------------------------------------------ ctypes types

class NDArrayDesc(ctypes.Structure):
    '''BNNSNDArrayDescriptor (bnns_structures.h).'''
    _fields_ = [('flags', ctypes.c_uint32),
                ('layout', ctypes.c_uint32),
                ('size', ctypes.c_size_t * 8),
                ('stride', ctypes.c_size_t * 8),
                ('data', ctypes.c_void_p),
                ('data_type', ctypes.c_uint32),
                ('table_data', ctypes.c_void_p),
                ('table_data_type', ctypes.c_uint32),
                ('data_scale', ctypes.c_float),
                ('data_bias', ctypes.c_float)]


class Activation(ctypes.Structure):
    '''BNNSActivation.'''
    _fields_ = [('function', ctypes.c_uint32),
                ('alpha', ctypes.c_float),
                ('beta', ctypes.c_float),
                ('iscale', ctypes.c_int32),
                ('ioffset', ctypes.c_int32),
                ('ishift', ctypes.c_int32),
                ('iscale_per_channel', ctypes.c_void_p),
                ('ioffset_per_channel', ctypes.c_void_p),
                ('ishift_per_channel', ctypes.c_void_p)]


class FCParams(ctypes.Structure):
    '''BNNSLayerParametersFullyConnected.'''
    _fields_ = [('i_desc', NDArrayDesc),
                ('w_desc', NDArrayDesc),
                ('o_desc', NDArrayDesc),
                ('bias', NDArrayDesc),
                ('activation', Activation)]


ALLOC_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                            ctypes.c_size_t, ctypes.c_size_t)
FREE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class FilterParams(ctypes.Structure):
    '''BNNSFilterParameters.'''
    _fields_ = [('flags', ctypes.c_uint32),
                ('n_threads', ctypes.c_size_t),
                ('alloc_memory', ALLOC_FN),
                ('free_memory', FREE_FN)]


class VImageBuffer(ctypes.Structure):
    '''vImage_Buffer.'''
    _fields_ = [('data', ctypes.c_void_p),
                ('height', ctypes.c_ulong),
                ('width', ctypes.c_ulong),
                ('rowBytes', ctypes.c_size_t)]


# Offsets a C compiler gives these structs on LP64 (checked on the host with
# gcc against the header definitions; see docs/native16_feasibility.md).
EXPECTED_LAYOUT = {
    'BNNSNDArrayDescriptor': (176, {'data': 136, 'data_type': 144,
                                    'table_data': 152, 'data_scale': 164}),
    'BNNSActivation': (48, {'iscale_per_channel': 24}),
    'BNNSLayerParametersFullyConnected': (752, {'w_desc': 176,
                                                'o_desc': 352,
                                                'bias': 528,
                                                'activation': 704}),
    'BNNSFilterParameters': (32, {'n_threads': 8, 'alloc_memory': 16}),
    'vImage_Buffer': (32, {'rowBytes': 24}),
}


def struct_layout():
    out = {}
    for name, cls in [('BNNSNDArrayDescriptor', NDArrayDesc),
                      ('BNNSActivation', Activation),
                      ('BNNSLayerParametersFullyConnected', FCParams),
                      ('BNNSFilterParameters', FilterParams),
                      ('vImage_Buffer', VImageBuffer)]:
        size, offs = EXPECTED_LAYOUT[name]
        got = {f: getattr(cls, f).offset for f in offs}
        out[name] = {'size': ctypes.sizeof(cls), 'expected_size': size,
                     'offsets': got, 'expected_offsets': offs,
                     'ok': ctypes.sizeof(cls) == size and got == offs}
    return out


# ------------------------------------------------------------- environment

def _sysctl(lib, name, want_str):
    size = ctypes.c_size_t(0)
    if lib.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0):
        return None
    buf = ctypes.create_string_buffer(max(size.value, 8))
    size = ctypes.c_size_t(len(buf))
    if lib.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0):
        return None
    raw = buf.raw[:size.value]
    if want_str:
        return raw.split(b'\0', 1)[0].decode('utf-8', 'replace')
    if len(raw) == 4:
        return struct.unpack('<i', raw)[0]
    if len(raw) == 8:
        return struct.unpack('<q', raw)[0]
    return raw.hex()


def sysctl_info():
    try:
        lib = ctypes.CDLL(None)
        fn = lib.sysctlbyname
    except (OSError, AttributeError):
        return {'available': False}
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                   ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                   ctypes.c_size_t]
    out = {'available': True}
    for n in SYSCTLS_STR:
        out[n] = _sysctl(lib, n, True)
    for n in SYSCTLS_INT:
        out[n] = _sysctl(lib, n, False)
    return out


def ios_info():
    '''iOS version / device via objc_util (Pythonista only).'''
    out = {}
    try:
        import objc_util
    except ImportError:
        return {'objc_util': False}
    out['objc_util'] = True
    try:
        dev = objc_util.ObjCClass('UIDevice').currentDevice()
        out['systemName'] = str(dev.systemName())
        out['systemVersion'] = str(dev.systemVersion())
        out['model'] = str(dev.model())
    except Exception as e:
        out['uidevice_error'] = repr(e)
    try:
        pi = objc_util.ObjCClass('NSProcessInfo').processInfo()
        out['operatingSystemVersionString'] = str(
            pi.operatingSystemVersionString())
        out['thermalState'] = int(pi.thermalState())
        out['lowPowerMode'] = bool(pi.isLowPowerModeEnabled())
        out['physicalMemory'] = int(pi.physicalMemory())
    except Exception as e:
        out['processinfo_error'] = repr(e)
    return out


def _task_footprint():
    '''phys_footprint in bytes (macOS/iOS), else None. As in hf_run.py.'''
    if sys.platform not in ('darwin', 'ios'):
        return None
    u64 = ctypes.c_uint64

    class TaskVMInfo(ctypes.Structure):
        _fields_ = ([('virtual_size', u64), ('region_count', ctypes.c_int32),
                     ('page_size', ctypes.c_int32)]
                    + [(n, u64) for n in (
                        'resident_size', 'resident_size_peak', 'device',
                        'device_peak', 'internal', 'internal_peak',
                        'external', 'external_peak', 'reusable',
                        'reusable_peak', 'purgeable_volatile_pmap',
                        'purgeable_volatile_resident',
                        'purgeable_volatile_virtual', 'compressed',
                        'compressed_peak', 'compressed_lifetime',
                        'phys_footprint', 'min_address', 'max_address')])
    try:
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        task = ctypes.c_uint32.in_dll(lib, 'mach_task_self_')
        task_info = lib.task_info
    except (OSError, ValueError, AttributeError):
        return None
    task_info.restype = ctypes.c_int
    task_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
                          ctypes.POINTER(ctypes.c_uint32)]
    info = TaskVMInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
    if task_info(task.value, 22, ctypes.byref(info), ctypes.byref(count)):
        return None
    if count.value * 4 < TaskVMInfo.min_address.offset:
        return None
    return info.phys_footprint


def alignment_info():
    out = {}
    if np is None:
        return out
    a = [np.empty(n, np.float32).ctypes.data for n in (17, 4096, 1 << 20)]
    b = [np.empty(n, np.uint16).ctypes.data for n in (17, 4096, 1 << 20)]
    out['np_empty_f32_addr_mod_64'] = [x % 64 for x in a]
    out['np_empty_u16_addr_mod_64'] = [x % 64 for x in b]
    try:
        out['mmap_page_size'] = __import__('mmap').PAGESIZE
    except Exception:
        pass
    return out


def environment():
    env = {
        'tool_version': TOOL_VERSION,
        'time': datetime.datetime.now().isoformat(),
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'sys_platform': sys.platform,
        'byteorder': sys.byteorder,
        'pointer_bytes': ctypes.sizeof(ctypes.c_void_p),
        'size_t_bytes': ctypes.sizeof(ctypes.c_size_t),
        'long_bytes': ctypes.sizeof(ctypes.c_long),
        'numpy': np.__version__ if np is not None else None,
        'repo': REPO,
        'repo_git_head': _git_head(),
    }
    try:
        env['uname'] = list(os.uname())
    except AttributeError:
        pass
    env['ios'] = ios_info()
    env['pythonista'] = bool(env['ios'].get('objc_util'))
    env['sysctl'] = sysctl_info()
    env['alignment'] = alignment_info()
    env['footprint_bytes'] = _task_footprint()
    env['struct_layout'] = struct_layout()
    env['tinygpt_backend'] = _backend_info()
    return env


def _git_head():
    try:
        with open(os.path.join(REPO, '.git', 'HEAD')) as f:
            head = f.read().strip()
        if head.startswith('ref: '):
            p = os.path.join(REPO, '.git', head[5:])
            if os.path.exists(p):
                with open(p) as f:
                    return head[5:] + ' ' + f.read().strip()
        return head
    except OSError:
        return None


def _backend_info():
    try:
        sys.path.insert(0, REPO)
        from tinygpt import backend
        return {'HAVE_ACCELERATE': backend.HAVE_ACCELERATE,
                'HAVE_SGEMV': backend.HAVE_SGEMV,
                'HAVE_VFORCE': backend.HAVE_VFORCE,
                'lib_path': backend._LIB_PATH}
    except Exception as e:
        return {'error': repr(e)}
    finally:
        if sys.path and sys.path[0] == REPO:
            sys.path.pop(0)


# ----------------------------------------------------------------- symbols

def load_libraries():
    libs, info = {}, {}
    for name, path in LIBRARIES:
        try:
            libs[name] = ctypes.CDLL(path)
            info[name] = {'path': path, 'loaded': True}
        except OSError as e:
            info[name] = {'path': path, 'loaded': False, 'error': str(e)}
    return libs, info


def symbol_table(libs):
    out = []
    for sym, status in SYMBOLS:
        where = []
        for name, _ in LIBRARIES:
            lib = libs.get(name)
            if lib is None:
                continue
            try:
                getattr(lib, sym)      # dlsym only; nothing is called
                where.append(name)
            except AttributeError:
                pass
        out.append({'symbol': sym, 'header_status': status,
                    'present': bool(where), 'found_in': where})
    return out


# ------------------------------------------------------------------- probe

def family(step):
    '''Steps that share a crash fate: one FC/MatMul dtype combination
    (all shapes and flags), or the step itself.'''
    if step.startswith(('FC_', 'MatMul_')):
        return '_'.join(step.split('_')[:4])
    return step


class Progress(object):
    '''Crash breadcrumbs: the step about to run is on disk before it runs.

    A leftover progress file means the previous run died in that step. It
    is added to a persistent crash list (next to it), and every step in
    the same family is skipped from then on, so a crashing dtype
    combination costs one restart, not one per shape.
    '''

    def __init__(self, path):
        self.path = path
        self.crash_path = os.path.join(os.path.dirname(path),
                                       'native16_diag_crashes.json')
        self.crashes = []
        self.new_crash = None
        if os.path.exists(self.crash_path):
            try:
                with open(self.crash_path) as f:
                    self.crashes = list(json.load(f))
            except (OSError, ValueError):
                self.crashes = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    step = json.load(f).get('step')
            except (OSError, ValueError):
                step = '<unreadable progress file>'
            self.new_crash = step
            if step not in self.crashes:
                self.crashes.append(step)
            with open(self.crash_path, 'w') as f:
                json.dump(self.crashes, f)

    def crashed_step(self):
        return self.new_crash

    def skipped(self, step):
        fams = set(family(c) for c in self.crashes)
        return family(step) in fams

    def mark(self, step):
        with open(self.path, 'w') as f:
            json.dump({'step': step, 'time': time.time()}, f)
            f.flush()
            os.fsync(f.fileno())

    def done(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


class NullTracker(object):
    '''Stand-in when the allocation hook is unusable: BNNS allocates for
    itself and the byte counts are reported as None (unmeasured).'''
    events = ()

    def mark(self):
        return 0

    def allocated_since(self, mark):
        return None

    def params(self, flags=0):
        p = FilterParams()
        p.flags = flags
        return p


class AllocTracker(object):
    '''Implements BNNSFilterParameters.alloc_memory / free_memory with
    libc posix_memalign / free and records every request.'''

    def __init__(self, libc):
        self._pm = libc.posix_memalign
        self._pm.restype = ctypes.c_int
        self._pm.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                             ctypes.c_size_t]
        self._free = libc.free
        self._free.restype = None
        self._free.argtypes = [ctypes.c_void_p]
        self.events = []
        self.live = {}
        # The callback objects must outlive every filter created with them.
        self.alloc_cb = ALLOC_FN(self._alloc)
        self.free_cb = FREE_FN(self._do_free)

    def _alloc(self, memptr, alignment, size):
        align = max(int(alignment), ctypes.sizeof(ctypes.c_void_p))
        rc = self._pm(memptr, align, max(int(size), 1))
        if rc == 0:
            self.live[memptr[0]] = int(size)
            self.events.append(('alloc', int(size)))
        return rc

    def _do_free(self, ptr):
        if ptr:
            self.events.append(('free', self.live.pop(ptr, -1)))
            self._free(ptr)

    def mark(self):
        return len(self.events)

    def allocated_since(self, mark):
        ev = self.events[mark:]
        sizes = [s for k, s in ev if k == 'alloc']
        return {'count': len(sizes), 'bytes': sum(sizes),
                'largest': max(sizes) if sizes else 0}

    def params(self, flags=0):
        p = FilterParams()
        p.flags = flags
        p.n_threads = 0
        p.alloc_memory = self.alloc_cb
        p.free_memory = self.free_cb
        return p


def _bind(lib, name, restype, argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = argtypes
    return f


def _vec_desc(arr, dtype):
    d = NDArrayDesc()
    d.layout = LAYOUT_VECTOR
    d.size[0] = arr.size
    d.data = arr.ctypes.data
    d.data_type = dtype
    return d


def _np_storage(dtype):
    return {DT_F32: np.float32, DT_F16: np.float16, DT_BF16: np.uint16}[dtype]


def f32_to_bf16_bits(a):
    '''Round-to-nearest-even float32 -> bfloat16 bits (finite inputs).'''
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def encode(a32, dtype):
    '''float32 values -> (storage array, exactly representable float64).'''
    if dtype == DT_F32:
        s = np.ascontiguousarray(a32, np.float32)
        return s, s.astype(np.float64)
    if dtype == DT_F16:
        s = a32.astype(np.float16)
        return s, s.astype(np.float64)
    s = f32_to_bf16_bits(a32)
    return s, bf16_bits_to_f32(s).astype(np.float64)


def decode(arr, dtype):
    if dtype == DT_BF16:
        return bf16_bits_to_f32(arr).astype(np.float64)
    return arr.astype(np.float64)


def errors(got, ref):
    d = got - ref
    scale = float(np.abs(ref).max()) + 1e-30
    return {'max_abs': float(np.abs(d).max()),
            'rms': float(np.sqrt(np.mean(d * d))),
            'max_rel_to_max': float(np.abs(d).max()) / scale,
            'finite': bool(np.isfinite(got).all())}


def flip_sign_bits(storage, dtype):
    '''Negate every weight in place (sign bit is bit 15 for F16 and BF16).'''
    if dtype == DT_F32:
        storage *= -1
    else:
        v = storage.view(np.uint16)
        v ^= np.uint16(0x8000)


class Prober(object):

    def __init__(self, libs, progress, skip, log):
        self.libs = libs
        self.progress = progress
        self.skip = set(skip)
        self.log = log
        self.acc = libs.get('Accelerate')
        self.results = {}
        self.rng = np.random.RandomState(1234)
        try:
            self.libc = ctypes.CDLL(None)
            self.tracker = AllocTracker(self.libc)
        except (OSError, AttributeError) as e:
            self.tracker = NullTracker()
            self.results['alloc_tracker_error'] = repr(e)
        if progress.skipped('alloc_hook_selftest'):
            # The Python allocation callback itself crashed once: run the
            # remaining probes with BNNS's own allocator (no byte counts).
            self.tracker = NullTracker()
            self.results['alloc_tracker_error'] = 'disabled after a crash'

    def step(self, name, fn):
        if name in self.skip or self.progress.skipped(name):
            self.results[name] = {'status': 'skipped',
                                  'reason': 'crashed in a previous run or '
                                            '--skip'}
            self.log('  %-44s skipped' % name)
            return None
        self.progress.mark(name)
        try:
            r = fn()
        except Exception as e:
            r = {'status': 'python-error', 'error': repr(e),
                 'trace': traceback.format_exc(limit=3)}
        self.results[name] = r
        self.log('  %-44s %s' % (name, _short(r)))
        return r

    def fn(self, name, restype, argtypes):
        try:
            return _bind(self.acc, name, restype, argtypes)
        except AttributeError:
            return None

    # -- individual probes -------------------------------------------------

    def get_data_size(self):
        f = self.fn('BNNSNDArrayGetDataSize', ctypes.c_size_t,
                    [ctypes.POINTER(NDArrayDesc)])
        if f is None:
            return {'status': 'unavailable'}
        out = {'status': 'ok'}
        for dt in (DT_F32, DT_F16, DT_BF16):
            d = NDArrayDesc()
            d.layout = LAYOUT_ROW_MAJOR
            d.size[0], d.size[1] = 5, 3
            d.data_type = dt
            got = int(f(ctypes.byref(d)))
            want = 15 * DTYPE_BYTES[dt]
            out[DTYPE_NAMES[dt]] = {'got': got, 'expected': want}
            if got != want:
                out['status'] = 'mismatch'
        return out

    def bnns_copy(self, src_dt):
        f = self.fn('BNNSCopy', ctypes.c_int,
                    [ctypes.POINTER(NDArrayDesc), ctypes.POINTER(NDArrayDesc),
                     ctypes.c_void_p])
        if f is None:
            return {'status': 'unavailable'}
        vals = self.rng.standard_normal(1027).astype(np.float32)
        src, exact = encode(vals, src_dt)
        dst = np.full(src.size, np.nan, np.float32)
        rc = f(ctypes.byref(_vec_desc(dst, DT_F32)),
               ctypes.byref(_vec_desc(src, src_dt)), None)
        if rc != 0:
            return {'status': 'refused', 'rc': int(rc)}
        e = errors(dst.astype(np.float64), exact)
        return {'status': 'ok' if e['max_abs'] == 0 else 'inexact',
                'rc': int(rc), 'err': e}

    def vimage_f16(self):
        f = self.fn('vImageConvert_Planar16FtoPlanarF', ctypes.c_ssize_t,
                    [ctypes.POINTER(VImageBuffer),
                     ctypes.POINTER(VImageBuffer), ctypes.c_uint32])
        if f is None:
            return {'status': 'unavailable'}
        vals = self.rng.standard_normal(1027).astype(np.float32)
        src, exact = encode(vals, DT_F16)
        dst = np.full(src.size, np.nan, np.float32)
        s = VImageBuffer(src.ctypes.data, 1, src.size, src.nbytes)
        d = VImageBuffer(dst.ctypes.data, 1, dst.size, dst.nbytes)
        rc = f(ctypes.byref(s), ctypes.byref(d), 0)
        if rc != 0:
            return {'status': 'refused', 'rc': int(rc)}
        e = errors(dst.astype(np.float64), exact)
        return {'status': 'ok' if e['max_abs'] == 0 else 'inexact',
                'rc': int(rc), 'err': e}

    def fully_connected(self, w_dt, i_dt, o_dt, n_out, n_in, client_ptr):
        create = self.fn('BNNSFilterCreateLayerFullyConnected',
                         ctypes.c_void_p,
                         [ctypes.POINTER(FCParams),
                          ctypes.POINTER(FilterParams)])
        apply_ = self.fn('BNNSFilterApply', ctypes.c_int,
                         [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
        destroy = self.fn('BNNSFilterDestroy', None, [ctypes.c_void_p])
        if None in (create, apply_, destroy):
            return {'status': 'unavailable'}
        W32 = (self.rng.standard_normal((n_out, n_in)) /
               np.sqrt(n_in)).astype(np.float32)
        x32 = self.rng.standard_normal(n_in).astype(np.float32)
        Ws, Wx = encode(W32, w_dt)
        xs, xx = encode(x32, i_dt)
        Ws = np.ascontiguousarray(Ws)
        y = np.zeros(n_out, _np_storage(o_dt))
        ref = Wx @ xx

        p = FCParams()
        p.i_desc = _vec_desc(xs, i_dt)
        p.i_desc.data = None
        p.o_desc = _vec_desc(y, o_dt)
        p.o_desc.data = None
        w = NDArrayDesc()
        w.layout = LAYOUT_ROW_MAJOR
        w.size[0], w.size[1] = n_in, n_out     # columns, rows
        w.data = Ws.ctypes.data
        w.data_type = w_dt
        p.w_desc = w
        p.bias.data = None
        p.bias.data_type = o_dt
        p.bias.layout = LAYOUT_VECTOR
        p.bias.size[0] = n_out
        p.activation.function = ACT_IDENTITY
        fp = self.tracker.params(FLAG_USE_CLIENT_PTR if client_ptr else 0)

        m0 = self.tracker.mark()
        filt = create(ctypes.byref(p), ctypes.byref(fp))
        out = {'shape': [n_out, n_in], 'client_ptr': client_ptr,
               'weight_bytes': int(Ws.nbytes),
               'create_alloc': self.tracker.allocated_since(m0)}
        if not filt:
            out['status'] = 'refused'
            return out
        try:
            m1 = self.tracker.mark()
            rc = apply_(filt, xs.ctypes.data, y.ctypes.data)
            out['apply_rc'] = int(rc)
            out['first_apply_alloc'] = self.tracker.allocated_since(m1)
            if rc != 0:
                out['status'] = 'apply-failed'
                return out
            got = decode(y, o_dt)
            out['err'] = errors(got, ref)
            # Weight transposed by mistake would give garbage, not a small
            # error; report the comparison so a layout mixup is visible.
            m2 = self.tracker.mark()
            rc = apply_(filt, xs.ctypes.data, y.ctypes.data)
            out['second_apply_alloc'] = self.tracker.allocated_since(m2)
            # Does the filter still read *our* buffer? Negate the weights in
            # place and apply again.
            flip_sign_bits(Ws, w_dt)
            rc2 = apply_(filt, xs.ctypes.data, y.ctypes.data)
            flip_sign_bits(Ws, w_dt)
            if rc2 == 0:
                got2 = decode(y, o_dt)
                tol = 4 * out['err']['max_abs'] + 1e-6
                if np.abs(got2 + got).max() <= tol:
                    out['reads_client_weights_after_create'] = True
                elif np.abs(got2 - got).max() <= tol:
                    out['reads_client_weights_after_create'] = False
                else:
                    out['reads_client_weights_after_create'] = 'unclear'
            tol = _fc_tolerance(w_dt, i_dt, o_dt, n_in)
            out['tolerance'] = tol
            out['status'] = ('ok' if out['err']['finite']
                             and out['err']['max_rel_to_max'] <= tol
                             else 'wrong-result')
            return out
        finally:
            m3 = self.tracker.mark()
            destroy(filt)
            if isinstance(self.tracker, AllocTracker):
                freed = [s for k, s in self.tracker.events[m3:]
                         if k == 'free']
                out['destroy_freed_bytes'] = sum(s for s in freed if s > 0)

    def matmul(self, w_dt, i_dt, o_dt, n_out, n_in):
        ws_fn = self.fn('BNNSMatMulWorkspaceSize', ctypes.c_ssize_t,
                        [ctypes.c_bool, ctypes.c_bool, ctypes.c_float,
                         ctypes.POINTER(NDArrayDesc),
                         ctypes.POINTER(NDArrayDesc),
                         ctypes.POINTER(NDArrayDesc),
                         ctypes.POINTER(FilterParams)])
        mm = self.fn('BNNSMatMul', ctypes.c_int,
                     [ctypes.c_bool, ctypes.c_bool, ctypes.c_float,
                      ctypes.POINTER(NDArrayDesc), ctypes.POINTER(NDArrayDesc),
                      ctypes.POINTER(NDArrayDesc), ctypes.c_void_p,
                      ctypes.POINTER(FilterParams)])
        if ws_fn is None or mm is None:
            return {'status': 'unavailable'}
        W32 = (self.rng.standard_normal((n_out, n_in)) /
               np.sqrt(n_in)).astype(np.float32)
        x32 = self.rng.standard_normal(n_in).astype(np.float32)
        Ws, Wx = encode(W32, w_dt)
        xs, xx = encode(x32, i_dt)
        Ws = np.ascontiguousarray(Ws)
        y = np.zeros(n_out, _np_storage(o_dt))
        ref = Wx @ xx

        def mat(arr, rows, cols, dt):
            # 2DFirstMajor: (i, j) at j*stride[0] + i*stride[1]; row-major.
            d = NDArrayDesc()
            d.layout = LAYOUT_2D_FIRST_MAJOR
            d.size[0], d.size[1] = rows, cols
            d.stride[0], d.stride[1] = 1, cols
            d.data = arr.ctypes.data
            d.data_type = dt
            return d

        A = mat(Ws, n_out, n_in, w_dt)
        B = mat(xs, n_in, 1, i_dt)
        C = mat(y, n_out, 1, o_dt)
        fp = self.tracker.params(0)
        out = {'shape': [n_out, n_in], 'weight_bytes': int(Ws.nbytes)}
        ws = int(ws_fn(False, False, 1.0, ctypes.byref(A), ctypes.byref(B),
                       ctypes.byref(C), ctypes.byref(fp)))
        out['workspace_bytes'] = ws
        if ws < 0:
            out['status'] = 'refused'
            return out
        work = np.empty(max(ws, 1), np.uint8)
        m0 = self.tracker.mark()
        rc = mm(False, False, 1.0, ctypes.byref(A), ctypes.byref(B),
                ctypes.byref(C), work.ctypes.data if ws else None,
                ctypes.byref(fp))
        out['rc'] = int(rc)
        out['call_alloc'] = self.tracker.allocated_since(m0)
        if rc != 0:
            out['status'] = 'apply-failed'
            return out
        out['err'] = errors(decode(y, o_dt), ref)
        tol = _fc_tolerance(w_dt, i_dt, o_dt, n_in)
        out['tolerance'] = tol
        out['status'] = ('ok' if out['err']['finite']
                         and out['err']['max_rel_to_max'] <= tol
                         else 'wrong-result')
        return out

    # -- driver ------------------------------------------------------------

    def run(self):
        self.step('struct_layout', lambda: {
            'status': 'ok' if all(v['ok'] for v in struct_layout().values())
            else 'mismatch'})
        if self.results['struct_layout'].get('status') != 'ok':
            self.log('  struct layout mismatch: no BNNS calls made')
            return self.results
        r = self.step('BNNSNDArrayGetDataSize', self.get_data_size)
        if r and r.get('status') == 'mismatch':
            self.log('  descriptor size mismatch: no further BNNS calls')
            return self.results
        # The allocation hook is a ctypes callback invoked from inside BNNS.
        # Exercise it once on its own so that, if callbacks misbehave on
        # this device, the crash is attributed to the hook rather than to
        # the first dtype combination.
        if isinstance(self.tracker, AllocTracker):
            self.step('alloc_hook_selftest', lambda: self.fully_connected(
                DT_F32, DT_F32, DT_F32, 8, 16, False))
        for dt in (DT_F16, DT_BF16):
            self.step('BNNSCopy_%s_to_F32' % DTYPE_NAMES[dt],
                      lambda dt=dt: self.bnns_copy(dt))
        self.step('vImage_Planar16F_to_PlanarF', self.vimage_f16)
        combos = [(DT_F32, DT_F32, DT_F32), (DT_F16, DT_F32, DT_F32),
                  (DT_BF16, DT_F32, DT_F32), (DT_F16, DT_F16, DT_F32),
                  (DT_F16, DT_F16, DT_F16), (DT_BF16, DT_BF16, DT_F32),
                  (DT_BF16, DT_BF16, DT_BF16)]
        shapes = [(8, 16), (40, 72), (256, 512)]
        for w, i, o in combos:
            tag = 'W%s_x%s_y%s' % (DTYPE_NAMES[w], DTYPE_NAMES[i],
                                   DTYPE_NAMES[o])
            for n_out, n_in in shapes:
                for cp in (True, False):
                    self.step('FC_%s_%dx%d_%s' % (tag, n_out, n_in,
                                                 'clientptr' if cp else 'copy'),
                              lambda w=w, i=i, o=o, a=n_out, b=n_in, cp=cp:
                              self.fully_connected(w, i, o, a, b, cp))
            for n_out, n_in in shapes:
                self.step('MatMul_%s_%dx%d' % (tag, n_out, n_in),
                          lambda w=w, i=i, o=o, a=n_out, b=n_in:
                          self.matmul(w, i, o, a, b))
        return self.results


def _fc_tolerance(w_dt, i_dt, o_dt, n_in):
    '''Acceptable max error relative to max |ref|.

    The reference uses the *exactly decoded* 16-bit weights and inputs, so
    storage rounding is excluded; what remains is accumulation and output
    rounding. Generous on purpose: this stage only asks "is it plausibly
    computing W @ x", not "is it accurate enough" (that is stage 2).
    '''
    lo = min(DTYPE_BYTES[w_dt], DTYPE_BYTES[i_dt], DTYPE_BYTES[o_dt])
    if o_dt == DT_BF16:
        return 2e-2
    if lo == 2 and (o_dt == DT_F16 or i_dt == DT_F16):
        return 5e-3
    return 1e-3


def _short(r):
    if not isinstance(r, dict):
        return str(r)
    s = r.get('status', '?')
    if 'err' in r:
        s += '  rel %.2g' % r['err']['max_rel_to_max']
    if 'reads_client_weights_after_create' in r:
        s += '  reads-client-W=%s' % r['reads_client_weights_after_create']
    if r.get('create_alloc'):
        s += '  create-alloc %dB' % r['create_alloc']['bytes']
    if r.get('first_apply_alloc'):
        s += '  apply-alloc %dB' % r['first_apply_alloc']['bytes']
    if 'workspace_bytes' in r:
        s += '  ws %dB' % r['workspace_bytes']
    return s


# ---------------------------------------------------------- model headers

def model_alignment(path):
    '''Byte alignment of each tensor in a local safetensors checkpoint
    (memory-mapped; reads headers, touches no weight pages).'''
    sys.path.insert(0, REPO)
    try:
        from tinygpt.hf.safetensors import Checkpoint
        ck = Checkpoint(path, use_mmap=True)
        try:
            out = {'tensors': 0, 'dtypes': {}, 'addr_mod_16_nonzero': 0,
                   'addr_mod_64_hist': {}, 'examples': []}
            for name in sorted(ck.keys()):
                t = ck.info(name)
                a = ck.get_raw(name).ctypes.data
                out['tensors'] += 1
                out['dtypes'][t.dtype] = out['dtypes'].get(t.dtype, 0) + 1
                k = str(a % 64)
                out['addr_mod_64_hist'][k] = out['addr_mod_64_hist'].get(k, 0) + 1
                if a % 16:
                    out['addr_mod_16_nonzero'] += 1
                if len(out['examples']) < 6 and len(t.shape) == 2:
                    out['examples'].append({'name': name, 'dtype': t.dtype,
                                            'shape': list(t.shape),
                                            'addr_mod_64': a % 64})
            return out
        finally:
            ck.close()
    except Exception as e:
        return {'error': repr(e)}
    finally:
        if sys.path and sys.path[0] == REPO:
            sys.path.pop(0)


# ------------------------------------------------------------------ report

def summarize(res):
    lines = []
    env = res['environment']
    sc = env.get('sysctl', {})
    ios = env.get('ios', {})
    lines.append('native16 diagnostics  %s  mode=%s' % (TOOL_VERSION,
                                                       res['mode']))
    lines.append('time      %s' % env['time'])
    lines.append('device    %s  %s  iOS %s' % (
        sc.get('hw.machine') or env.get('machine'), ios.get('model', ''),
        ios.get('systemVersion', sc.get('kern.osproductversion', '?'))))
    lines.append('os build  %s' % sc.get('kern.osversion'))
    lines.append('python    %s' % env['python'].split()[0])
    lines.append('numpy     %s   pointer %d bytes   pythonista %s' % (
        env['numpy'], env['pointer_bytes'], env['pythonista']))
    lines.append('cpu       FEAT_FP16=%s FEAT_BF16=%s FHM=%s I8MM=%s SME=%s '
                 'ncpu=%s perf-cores=%s' % (
                     sc.get('hw.optional.arm.FEAT_FP16'),
                     sc.get('hw.optional.arm.FEAT_BF16'),
                     sc.get('hw.optional.arm.FEAT_FHM'),
                     sc.get('hw.optional.arm.FEAT_I8MM'),
                     sc.get('hw.optional.arm.FEAT_SME'),
                     sc.get('hw.ncpu'), sc.get('hw.perflevel0.physicalcpu')))
    lines.append('caches    L2(P)=%s cacheline=%s mem=%s' % (
        sc.get('hw.perflevel0.l2cachesize') or sc.get('hw.l2cachesize'),
        sc.get('hw.cachelinesize'), sc.get('hw.memsize')))
    lines.append('structs   %s' % ('ok' if all(
        v['ok'] for v in env['struct_layout'].values()) else 'MISMATCH'))
    lines.append('')
    lines.append('libraries')
    for k, v in res['libraries'].items():
        lines.append('  %-10s %s' % (k, 'loaded' if v['loaded']
                                     else 'not loaded'))
    lines.append('')
    lines.append('symbols')
    for s in res['symbols']:
        lines.append('  %-38s %-8s %s' % (s['symbol'],
                                          'PRESENT' if s['present'] else '-',
                                          s['header_status']))
    if res.get('previous_crash'):
        lines.append('')
        lines.append('PREVIOUS RUN DIED DURING: %s (its family is skipped '
                     'from now on)' % res['previous_crash'])
    if res.get('crash_history'):
        lines.append('crash history: %s' % ', '.join(res['crash_history']))
    if 'probe' in res:
        lines.append('')
        lines.append('probe results')
        for k, v in res['probe'].items():
            lines.append('  %-44s %s' % (k, _short(v)))
    if 'model' in res:
        lines.append('')
        lines.append('model alignment  %s' % json.dumps(res['model']))
    return '\n'.join(lines) + '\n'


def run(mode='symbols', model=None, out_dir=None, skip=(), log=print):
    out_dir = out_dir or os.path.join(REPO, 'native16_results')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    progress = Progress(os.path.join(out_dir, 'native16_diag_progress.json'))
    res = {'mode': mode, 'environment': environment()}
    crashed = progress.crashed_step()
    if crashed:
        res['previous_crash'] = crashed
        log('Previous run did not finish; it died during %r. Skipping it '
            'and its dtype family from now on.' % crashed)
    res['crash_history'] = list(progress.crashes)
    libs, res['libraries'] = load_libraries()
    res['symbols'] = symbol_table(libs)
    if model:
        res['model'] = model_alignment(model)
    if mode == 'probe':
        if np is None:
            res['probe'] = {'status': 'numpy missing'}
        elif 'Accelerate' not in libs:
            res['probe'] = {'status': 'Accelerate not loadable; nothing to '
                                      'probe on this platform'}
        else:
            log('probing (tiny matrices; a crash here is reported on the '
                'next run):')
            res['probe'] = Prober(libs, progress, skip, log).run()
    res['footprint_bytes_end'] = _task_footprint()
    progress.done()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.join(out_dir, 'diag_%s_%s' % (mode, stamp))
    with open(base + '.json', 'w') as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    text = summarize(res)
    with open(base + '.txt', 'w') as f:
        f.write(text)
    return res, text, base


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--probe', action='store_true',
                    help='also call documented BNNS/vImage functions on '
                         'tiny matrices')
    ap.add_argument('--model', help='local model directory to check tensor '
                                    'alignment (headers only)')
    ap.add_argument('--out', help='results directory '
                                  '(default <repo>/native16_results)')
    ap.add_argument('--skip', action='append', default=[],
                    help='probe step name to skip (repeatable)')
    a = ap.parse_args(argv)
    res, text, base = run('probe' if a.probe else 'symbols', a.model, a.out,
                          a.skip)
    print(text)
    print('wrote %s.json and %s.txt' % (base, base))
    return 0


def find_models(root=None, limit=5):
    '''Model folders (with model.safetensors[.index.json]) under hf_models/.'''
    root = root or os.path.join(REPO, 'hf_models')
    out = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        if d[len(root):].count(os.sep) >= 5:
            dirs[:] = []
        if 'model.safetensors' in files or \
                'model.safetensors.index.json' in files:
            out.append(d)
            if len(out) >= limit:
                break
    return out


def interactive():
    models = find_models()
    model = models[0] if models else None
    res, text, base = run('symbols', model=model)
    print(text)
    print('wrote %s.txt' % base)
    try:
        ans = input('\nRun the probe stage too (tiny BNNS calls)? [y/N] ')
    except EOFError:
        ans = ''
    if ans.strip().lower().startswith('y'):
        res, text, base = run('probe')
        print(text)
        print('wrote %s.txt and .json -- please send both files back.' % base)


if __name__ == '__main__':
    if len(sys.argv) == 1:      # Pythonista's Run button passes no args
        interactive()
    else:
        sys.exit(main())
