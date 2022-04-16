# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility

__authors__ = ["Vincent Favre-Nicolin", "Ondrej Mandula"]
__license__ = "CeCILL-B"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2022-02-06"
__docformat__ = 'restructuredtext'
# Valid numbering includes 3.1, 3.1.0, 3.1.2, 3.1dev0, 3.1a0, 3.1b0
__version__ = "2022.1"


def get_git_version():
    """
    Get the full version name with git hash, e.g. "2020.1-65-g958b7254-dirty"
    Only works if the current directory is part of the git repository.
    :return: the version name
    """
    from subprocess import Popen, PIPE
    try:
        p = Popen(['git', 'describe', '--tags', '--dirty', '--always'],
                  stdout=PIPE, stderr=PIPE)
        return p.stdout.readlines()[0].strip().decode("UTF-8")
    except:
        # in distributed & installed versions this is replaced by a string
        __git_version_static__ = "git_version_placeholder"
        if "placeholder" in __git_version_static__:
            return __version__
        return __git_version_static__


def get_git_date():
    """
    Get the last git commit date, e.g. "2021-09-23T14:45:26+02:00"
    Only works if the current directory is part of the git repository.
    This can be interpreted e.g. using:
        datetime.datetime.fromisoformat(get_git_date())

    :return: the date string
    """
    from subprocess import Popen, PIPE
    try:
        p = Popen(['git', 'show', '-s', '--format=format:%cI'],
                  stdout=PIPE, stderr=PIPE)
        return p.stdout.readlines()[0].strip().decode("UTF-8")
    except:
        # in distributed & installed versions this is replaced by a string
        __git_date_static__ = "git_date_placeholder"
        if "placeholder" in __git_date_static__:
            return __date__
        return __git_date_static__
