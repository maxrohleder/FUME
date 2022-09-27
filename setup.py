from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fume_torch',
    version='2022.09.27',
    author='Maximilian Rohleder',
    author_email='Maximilian.Rohleder@fau.de',
    description='Trainable Fundamental Matrix based Epipolar Image Translation Layer (PyTorch GPU)',
    url='https://github.com/maxrohleder/fume',
    ext_modules=[
        CUDAExtension('fume_cuda', [
            'csrc/image_translation.cpp',
            'csrc/image_translation_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
