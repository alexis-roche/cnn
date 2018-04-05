import os
import shutil
from glob import glob
   
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import system_info

# TODO. Check OpenCL ICDs:
# /etc/OpenCL/vendors/


def _double_encoding_version(fname):
    base, ext = os.path.splitext(fname)
    out = base + '_d' + ext
    f = open(os.path.join('cnn', out), 'w')
    f.writelines((l.replace('float ', 'double ').replace('float*', 'double*') for l in open(os.path.join('cnn', fname))))
    f.close()
    return out


def add_opencl_files(config, files):
    if isinstance(files, str):
        files = [files]
    else:
        files = list(files)
    for f in files:
        config.add_data_files(f)
        config.add_data_files(_double_encoding_version(f))

        
def configuration(parent_package='',top_path=None):
    # Create package configuration
    config = Configuration('cnn', parent_package, top_path)
    config.add_subpackage('tests')
    
    # Add Cython module extension
    config.add_include_dirs(config.name.replace('.', os.sep))
    info = system_info()
    opts = info.calc_extra_info()
    if info.cp.has_section('opencl'):
        info.section = 'opencl'
        config.add_include_dirs(info.get_include_dirs())
        opts['library_dirs'] = info.get_lib_dirs()
        opts['extra_link_args'] = ['-l%s' % s for s in info.get_libraries()]
    else:
        opencl_link_args = ['-lOpenCL']
    config.add_extension('_utils', sources=['_utils.pyx', 'utils.c', 'opencl_utils.c'], **opts)

    #Add OpenCL kernel files
    add_opencl_files(config, ['test1d.cl', 'convolve_image.cl', 'relu_max_pool_image.cl'])
    
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
