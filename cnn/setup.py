import os

# TODO. Check OpenCL ICDs:
# /etc/OpenCL/vendors/


get_info = lambda cp, lib, key: [s.strip() for s in cp.get(lib, key).split(',')]


def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import system_info

    config = Configuration('cnn', parent_package, top_path)
    config.add_subpackage('tests')
    
    # Get OpenCL configuration
    info = system_info()
    info.parse_config_files()
    cp = info.cp
    
    if cp.has_section('opencl'):
        opencl_include_dirs = get_info(cp, 'opencl', 'include_dirs')
        opencl_library_dirs = get_info(cp, 'opencl', 'library_dirs')
        opencl_link_args = ['-l%s' % s for s in get_info(cp, 'opencl', 'libraries')]
    else:
        opencl_include_dirs = []
        opencl_library_dirs = []
        opencl_link_args = ['-lOpenCL']

    print('Detected OpenCL configuration:')
    print(' include_dirs = %s' % opencl_include_dirs)
    print(' library_dirs = %s' % opencl_library_dirs)
    print(' link_args = %s' % opencl_link_args)

    # Add runtime module extension
    config.add_data_files('utils.cl')
    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_include_dirs(opencl_include_dirs)
    config.add_extension('_run', sources=['_run.pyx', 'run_utils.c'], extra_compile_args=['-O3'])
    config.add_extension('_opencl',
                         sources=['_opencl.pyx', 'utils.c'],
                         extra_compile_args=['-O3'],
                         library_dirs=opencl_library_dirs,
                         extra_link_args=opencl_link_args)
    
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
