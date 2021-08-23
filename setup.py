# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2019  Baidu.com, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
"""
Setup script.
Authors: xianghuisun(2357094733@qq.com)
Date:    2021/8/20 00:00:01
"""
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="nlp-basictasks",
    version="0.1.4",
    author="xianghuisun",
    author_email="2357094733@qq.com",
    description="a simple framework that can quickly build some basic NLP tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xianghuisun/nlp-basictasks",
    # packages=setuptools.find_packages(),
    #packages = ['basictasks', 
    #    'basictasks.readers', 
    #    'basictasks.modules', 
    #    'basictasks.heads',
    #    'basictasks.tasks', 
    #    'basictasks.evaluation'],
    #package_dir={'basictasks':'./basictasks',
    #             'basictasks.readers':'./basictasks/readers',
    #             'basictasks.modules':'./basictasks/modules',
    #             'basictasks.heads':'./basictasks/heads',
    #             'basictasks.tasks': './basictasks/tasks',
    #             'basictasks.evaluation': './basictasks/evaluation'},
    #package_dir={"": "nlp_basictasks"},
    #packages=setuptools.find_packages(where="nlp_basictasks"),
    packages=setuptools.find_packages(),
    platforms = "any",
    license='Apache 2.0',
    classifiers = [
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
          ],
    python_requires='>=3.0',
    install_requires = [
        'torch>=1.6.0',
        'torchvision',
        'tqdm',
        'scipy',
        'scikit-learn',
        'seqeval',
        'pandas'
    ]
)
