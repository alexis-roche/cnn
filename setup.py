
#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
version = '0.1dev'

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )
    config.add_subpackage('cnn')
    return config


def setup_package(**extra_args):
    from numpy.distutils.core import setup
    
    setup(configuration=configuration,
          name='cnn',
          version=version,
          description='CNN for image analysis',
          requires = ('numpy', 'keras'),
          **extra_args)


if __name__ == '__main__':
    setup_package()
