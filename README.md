# Harvard CS107 Final Project: SmartDiff

[![Build Status](https://travis-ci.com/SmartDiff/cs107-FinalProject.svg?branch=master)](https://travis-ci.com/SmartDiff/cs107-FinalProject)
[![codecov](https://codecov.io/gh/SmartDiff/cs107-FinalProject/branch/master/graph/badge.svg?token=9IKFVF8E1T)](https://codecov.io/gh/SmartDiff/cs107-FinalProject)

## no-GUI branch
[![Build Status](https://travis-ci.com/SmartDiff/cs107-FinalProject.svg?branch=no-GUI)](https://travis-ci.com/SmartDiff/cs107-FinalProject)
[![codecov](https://codecov.io/gh/SmartDiff/cs107-FinalProject/branch/no-GUI/graph/badge.svg?token=9IKFVF8E1T)](https://codecov.io/gh/SmartDiff/cs107-FinalProject)

**Group 25**

Members: Bianca Cordazzo, Tianlei He, Xincheng Tan, Yilan Wang

### How to Install 

    - **From Github:** The user first installs SmartDiff and calls the main script in the commandline and interacts with the Graphic User Interface (GUI). **There is no need to import or instantiate python objects.**
    
        ```python
            git clone https://github.com/SmartDiff/cs107-FinalProject.git
            cd /dir/to/SmartDiff
            python main.py
        ```
        
        PyQt5 should be automatically installed as it is specified in requirements.txt. In case the user gets "cannot find module PyQt5" in the command line, do the following to install PyQt5 (default 5.15.1)

        ```python
            pip install PyQt5
        ```

    - **From PyPI**: SmartDiff is also available on PyPI https://pypi.org/project/Smartdiff/0.0.16/ and all the dependency should be automatically installed via the command below.

        ```python
            pip install Smartdiff==0.0.16
            cd /dir/to/SmartDiff
            python main.py
        ```
