from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
            sources=['src/shocc.cpp'],
            extra_compile_args={'cxx': [],
                                'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
