import os

# TODO. Check OpenCL ICDs:
# /etc/OpenCL/vendors/


def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import system_info

    config = Configuration('cnn', parent_package, top_path)
    config.add_subpackage('tests')
    
    # Add cython module extension
    config.add_data_files('*.cl')
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

    config.add_extension('_utils',
                         sources=['_utils.pyx', 'utils.c', 'opencl_utils.c'],
                         **opts)
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
