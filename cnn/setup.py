import os

def configuration(parent_package='',top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration('cnn', parent_package, top_path)
    config.add_subpackage('tests')
    #config.add_subpackage('testing')

    config.add_include_dirs(config.name.replace('.', os.sep))
    config.add_extension('_run', sources=['_run.pyx', 'run_utils.c'], extra_compile_args=['-O3'])
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
