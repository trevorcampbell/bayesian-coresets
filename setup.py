from distutils.core import setup
from distutils.command.build_clib import build_clib
from distutils.command.install import install
from distutils import log

class build_so_first(install):
  def run(self):
    self.run_command("build_clib")
    return install.run(self)

class build_so(build_clib):
  def build_libraries(self, libraries):
    for (lib_name, build_info) in libraries:
      sources = build_info.get('sources')
      if sources is None or not isinstance(sources, (list, tuple)):
          raise DistutilsSetupError(
                 "in 'libraries' option (library '%s'), "
                 "'sources' must be present and must be "
                 "a list of source filenames" % lib_name)
      sources = list(sources)
      compile_flags = build_info.get('compile_flags')
      if (compile_flags is not None) and (not isinstance(compile_flags, (list, tuple))):
          raise DistutilsSetupError(
                 "in 'libraries' option (library '%s'), "
                 "'compile_flags' (if present) must be a list of strings" % lib_name)
      compile_flags = list(compile_flags)


      log.info("building '%s' library", lib_name)

      # First, compile the source code to object files in the library
      # directory.  (This should probably change to putting object
      # files in a temporary build directory.)
      macros = build_info.get('macros')
      include_dirs = build_info.get('include_dirs')
      objects = self.compiler.compile(sources,
                                      output_dir=self.build_temp,
                                      macros=macros,
                                      include_dirs=include_dirs,
                                      debug=self.debug,
                                      extra_postargs=compile_flags)

      # Now link the object files together into a shared obj.
      self.compiler.link_shared_object(objects, lib_name+'.so',
                                      output_dir='hilbertcoresets',
                                      debug=self.debug, 
                                      extra_postargs=compile_flags)

libcaptreec = ('libcaptreec', {'sources': ['hilbertcoresets/captree.cpp'], 'compile_flags': ['-O3', '--std=c++17']})

setup(
    name = 'hilbertcoresets',
    version='0.3',
    description="Hilbert coresets for approximate Bayesian inference",
    author='Trevor Campbell',
    author_email='tdjc@mit.edu',
    url='https://github.com/trevorcampbell/hilbert-coresets/',
    packages=['hilbertcoresets'],
    install_requires=['numpy', 'scipy', 'ctypes', 'collections', 'heapq'],
    keywords = ['Bayesian', 'inference', 'coreset', 'Hilbert', 'Frank-Wolfe', 'greedy', 'geodesic'],
    platforms='ALL',
    libraries=[libcaptreec],
    cmdclass={'install':build_so_first, 'build_clib':build_so},
    package_data={'hilbertcoresets': ['libcaptreec.so']}
)
