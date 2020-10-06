from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HEALPIX_DIR = '/home/pierfied/Downloads/Healpix_3.70_2020Jul23/Healpix_3.70'

setup(
    name='SHoCC',
    version='0.1',
    packages=['shocc'],
    url='https://github.com/pierfied/shocc',
    license='',
    author='Pier Fiedorowicz',
    author_email='pierfied@email.arizona.edu',
    description='Spherical Harmonics on CUDA Cards',
    ext_modules=[
        CUDAExtension(
            name='shocc.driver',
            sources=['src/shocc.cu', 'src/standard_transforms.cu'],
            include_dirs=[HEALPIX_DIR + '/include'],
            extra_objects=[HEALPIX_DIR + '/lib/libhealpix_cxx.a'],
            extra_compile_args={'cxx': [],
                                'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
