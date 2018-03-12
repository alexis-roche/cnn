import os

# TODO. Check OpenCL ICDs:
# /etc/OpenCL/vendors/intelocl64.icd


def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    #from numpy.distutils.system_info import system_info

    config = Configuration('cnn', parent_package, top_path)
    config.add_subpackage('tests')
    #config.add_subpackage('testing')

    config.add_data_files('utils.cl')
    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension('_run', sources=['_run.pyx', 'run_utils.c'], extra_compile_args=['-O3'])

    config.add_extension('_opencl',
                         sources=['_opencl.pyx', 'utils.c'],
                         extra_compile_args=['-O3'],
                         extra_link_args=['-lOpenCL'])
    
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
