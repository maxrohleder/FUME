from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fume_torch',
    version='2022.10.04',
    author='Maximilian Rohleder',
    author_email='Maximilian.Rohleder@fau.de',
    description='Trainable Fundamental Matrix based Epipolar Image Translation Layer (PyTorch GPU)',
    url='https://github.com/maxrohleder/fume',
    py_modules=['fume'],
    ext_modules=[
        CUDAExtension('fume_torch_lib', [
            'cuda/bind_fume.cpp',
            'cuda/image_translation.cpp',
            'cuda/image_translation_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
