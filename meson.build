project('TCanny', 'cpp', version: '0', meson_version: '>=1.3.0',
  default_options: ['cpp_std=c++17', 'buildtype=release', 'warning_level=2', 'b_lto=true'])

cc = meson.get_compiler('cpp')
fs = import('fs')
py = import('python').find_installation()
vs_include_path = get_option('vapoursynth_headers')
if vs_include_path == ''
  vs_include_path = run_command(py, '-c',
    'import os, vapoursynth as vs; print(os.path.join(os.path.dirname(vs.__file__), "include"))',
    check: true).stdout().strip()
endif
vapoursynth = dependency('vapoursynth', version: '>=80', required: false)
if not vapoursynth.found()
  vapoursynth = declare_dependency(
    include_directories: include_directories(vs_include_path),
    version: '80')
  message('Using VapourSynth headers from ' + vs_include_path)
endif

# CPU SIMD (cpu_simd.cpp) is Highway: a system libhwy if it is recent enough, else the pinned wrap, linked statically.
hwy = dependency('libhwy', version: '>=1.4.0', fallback: ['highway', 'hwy_dep'],
  default_options: ['default_library=static', 'contrib=disabled', 'examples=disabled', 'tests=disabled', 'test_standalone=true'])
sources = files('TCanny/TCanny.cpp', 'TCanny/cpu_reference.cpp', 'TCanny/cpu_simd.cpp')
project_cpp_args = []
if get_option('cuda')
  project_cpp_args += ['-DTCANNY_WITH_CUDA']
endif
if get_option('vulkan')
  project_cpp_args += ['-DTCANNY_WITH_VULKAN']
endif
if get_option('profiling')
  project_cpp_args += ['-DTCANNY_PROF']
endif
if cc.get_argument_syntax() == 'gcc'
  # No implicit FMA contraction: the scalar reference (opt=1) must round the same on every architecture (aarch64 GCC
  # and Clang contract a*b+c by default), and the SIMD kernels fuse only where they call MulAdd/MulSub.
  project_cpp_args += ['-fno-math-errno', '-fno-trapping-math', '-ffp-contract=off']
endif
add_project_arguments(project_cpp_args, language: 'cpp')
plugin_link_args = []
if get_option('cuda')
  nvcc = find_program('nvcc', required: true)
  if host_machine.system() == 'windows'
    # Meson has no getenv(): read CUDA_PATH through Python, else use the toolkit that holds the nvcc found above.
    cuda_path = run_command(py, '-c', 'import os; print(os.environ.get("CUDA_PATH", ""))', check: true).stdout().strip()
    if cuda_path == ''
      cuda_path = fs.parent(fs.parent(nvcc.full_path()))
    endif
    if cuda_path != '' and fs.exists(cuda_path / 'lib/x64/cudart.lib')
      plugin_link_args += [cuda_path / 'lib/x64/cudart.lib']
    else
      plugin_link_args += ['cudart.lib']
    endif
  elif fs.exists('/usr/local/cuda/lib64/libcudart.so')
    plugin_link_args += ['/usr/local/cuda/lib64/libcudart.so']
  elif fs.exists('/usr/local/cuda-13.3/lib64/libcudart.so')
    plugin_link_args += ['/usr/local/cuda-13.3/lib64/libcudart.so']
  else
    plugin_link_args += ['-lcudart']
  endif
  cuda_include_args = []
  # The Python package's pkg-config file can point at an older compatibility
  # include directory. Always pass the headers selected by Meson first so
  # VSVulkan4.h and VapourSynth4.h come from the same R80+ SDK.
  cuda_include_args += ['-I' + vs_include_path]
  pkgconfig_program = find_program('pkg-config', required: false)
  if pkgconfig_program.found()
    pkg_cflags = run_command(pkgconfig_program, '--cflags-only-I', 'vapoursynth', check: false).stdout().strip()
    foreach compile_arg : pkg_cflags.split()
      if compile_arg.startswith('-I')
        cuda_include_args += [compile_arg]
      endif
    endforeach
  endif
  vulkan_headers = get_option('vulkan_headers')
  if vulkan_headers == '' and fs.exists('/usr/include/vulkan/vulkan_core.h')
    vulkan_headers = '/usr/include'
  endif
  if vulkan_headers != ''
    cuda_include_args += ['-I' + vulkan_headers]
  endif
  cuda_host_flags = []
  if host_machine.system() == 'windows'
    # nvcc defaults the host compiler to /MT; use the same CRT as the rest of the build.
    vscrt = get_option('b_vscrt')
    if vscrt == 'from_buildtype' or vscrt == 'static_from_buildtype'
      vscrt = (vscrt.startswith('static') ? 'mt' : 'md') + (get_option('buildtype') == 'debug' ? 'd' : '')
    endif
    if vscrt != 'none'
      cuda_host_flags += ['-Xcompiler=/' + vscrt.to_upper()]
    endif
  else
    cuda_host_flags += ['-Xcompiler=-fPIC']
  endif
  if get_option('profiling')
    cuda_host_flags += ['-DTCANNY_PROF']
  endif
  cuda_object_name = host_machine.system() == 'windows' ? 'tcanny_cuda.obj' : 'tcanny_cuda.o'
  cuda_object = custom_target('tcanny_cuda_object',
    input: files('TCanny/cuda_backend.cu'),
    output: cuda_object_name,
    command: [nvcc, '-std=c++17', '-O3', '--fmad=false'] + cuda_include_args + cuda_host_flags + ['-c', '@INPUT@', '-o', '@OUTPUT@'])
  sources += cuda_object
  link_with = []
else
  sources += files('TCanny/cuda_stub.cpp')
  link_with = []
endif

if get_option('vulkan')
  sources += files('TCanny/gpu_vulkan.cpp')
  vulkan_dep = dependency('vulkan', required: false)
  if not vulkan_dep.found()
    message('Vulkan loader not linked: VapourSynth supplies dispatch through VSVulkan4.h')
  endif
  glslang = find_program('glslangValidator', required: false)
  if glslang.found()
    custom_target('tcanny_vulkan_spv',
      input: files('TCanny/shaders/tcanny_vulkan.comp.glsl'),
      output: 'tcanny_vulkan.spv',
      command: [glslang, '-V', '-S', 'comp', '@INPUT@', '-o', '@OUTPUT@'],
      build_by_default: true)
    custom_target('tcanny_pack_spv',
      input: files('TCanny/shaders/tcanny_pack.comp.glsl'),
      output: 'tcanny_pack.spv',
      command: [glslang, '-V', '-S', 'comp', '@INPUT@', '-o', '@OUTPUT@'],
      build_by_default: true)
    custom_target('tcanny_hysteresis_tile_spv',
      input: files('TCanny/shaders/tcanny_hysteresis_tile.comp.glsl'),
      output: 'tcanny_hysteresis_tile.spv',
      command: [glslang, '-V', '-S', 'comp', '@INPUT@', '-o', '@OUTPUT@'],
      build_by_default: true)
    custom_target('tcanny_hysteresis_spv',
      input: files('TCanny/shaders/tcanny_hysteresis.comp.glsl'),
      output: 'tcanny_hysteresis.spv',
      command: [glslang, '-V', '-S', 'comp', '@INPUT@', '-o', '@OUTPUT@'],
      build_by_default: true)
  endif
else
  sources += files('TCanny/gpu_stub.cpp')
endif

plugin_module = shared_module('tcannyx', sources,
  dependencies: [vapoursynth, hwy],
  include_directories: [include_directories('TCanny')] + (get_option('vulkan_headers') != '' ? [include_directories(get_option('vulkan_headers'))] : []),
  link_with: link_with,
  link_args: plugin_link_args,
  install: true,
  install_dir: get_option('libdir') / 'vapoursynth',
  gnu_symbol_visibility: 'hidden')

if get_option('benchmarks')
  run_target('tcanny-benchmark',
    command: [py, meson.project_source_root() / 'bench/bench_tcanny.py', '--plugin', plugin_module.full_path()])
endif

if get_option('tests')
  test_exe = executable('tcanny-algorithm-test', 'tests/algorithm_test.cpp',
    include_directories: include_directories('TCanny'))
  test('tcanny-algorithm', test_exe)
  direction_exe = executable('tcanny-direction-test', 'tests/direction_test.cpp',
    include_directories: include_directories('TCanny', 'tests'), dependencies: hwy)
  test('tcanny-direction', direction_exe, timeout: 300)
  test('tcanny-vs-integration', py,
    args: ['-B', meson.project_source_root() / 'tests/vs_integration.py', '--plugin', plugin_module.full_path()])
  test('tcanny-shader-twins', py,
    args: ['-B', meson.project_source_root() / 'tests/check_shader_twins.py'])
  test('tcanny-cpu-golden', py,
    args: ['-B', meson.project_source_root() / 'tests/cpu_golden.py', '--plugin', plugin_module.full_path()])
  test('tcanny-cpu-simd', py,
    args: ['-B', meson.project_source_root() / 'tests/cpu_simd_digest.py', '--plugin', plugin_module.full_path(), '--pinned'],
    timeout: 300)
endif

summary({'VapourSynth': vapoursynth.version(), 'Highway': hwy.version(), 'Vulkan': get_option('vulkan'), 'CUDA': get_option('cuda'), 'Tests': get_option('tests')}, section: 'TCanny features')

