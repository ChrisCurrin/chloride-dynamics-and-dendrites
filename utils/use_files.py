# coding=utf-8
# ::--------------------------------------------------------------------------------------------
# :: Script to compile dll from .mod files quickly
# :: V1.0
# :: Author: Christopher Brian Currin
# :: Set up:
# ::	1) Place script in folder with .mod files
# ::	2) run script to compile .mod files
# ::	2.1) changes in .mod files from working directory are reflected
# ::	2.2) push enter after compilation to place new .dll in working directory
# ::	2.2.1) ensure any nrn instances using the to-be-overwritten .dll are closed
# ::--------------------------------------------------------------------------------------------
#
# DEL /F /S /Q /A "c:\nrn\MOD\*"
#
# XCOPY *.mod c:\nrn\MOD /y
# :: /m only updated files copied
# :: /e all subdirectories too
# :: /y confirm all
# set "var=%cd%"
# cd /d c:\nrn\MOD
# :: The “/d” parameter is used to change the current drive to a specific folder from another disk volume.
#     c:\nrn/mingw/bin/sh c:\nrn/lib/mknrndll.sh /c\nrn
# XCOPY nrnmech.dll "%var%" /m /y
from __future__ import print_function

import glob
import sys
import platform
import argparse
import shutil
import os
from subprocess import Popen, PIPE

import functools


def create_dir(path):
    """

    :return: path of created directory
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    return path


def copy_file(src, dest):
    # From shutil docs:
    #   If dest is a directory, a file with the same basename as
    #   src is created (or overwritten) in the directory specified.
    try:
        shutil.copy2(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


def move_file(src, dest):
    try:
        copy_file(src, dest)
    except (IOError, shutil.Error):
        print("error copying file")
    finally:
        os.remove(src)


class cd(object):
    """Context manager for changing the current working directory"""

    def __init__(self, new_path, with_logs=True):
        self.new_path = os.path.expanduser(new_path)
        self.with_logs = with_logs

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.new_path)
        if self.with_logs:
            print("in " + self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        if self.with_logs:
            print("out " + self.new_path)
            print("in " + self.savedPath)


def nrnivmodl(tries_left=2, base_process=None, clean_after=True):
    compiler_output, err = None, None
    # mod_path = mod_path.replace("\\", "/")
    if tries_left == -1:
        # reached max tries
        return compiler_output, err
    if base_process is None:
        base_process = ['nrnivmodl']
        if platform.system() == 'Windows':
            # base_process = ["c:\\nrn/mingw/bin/sh", "c:\\nrn/lib/mknrndll.sh", "/c\\nrn"]
            base_process[0] += ".bat"
        # else:
        #     print("unknown system")
        #     sys.exit(-1)
    try:
        # start compiler process and monitor output
        process = Popen(base_process, stdin=PIPE, stdout=PIPE)
        compiler_output, err = process.communicate()
    except OSError as err:
        print("'{}' failed".format(base_process))
        alt_path = "/usr/local/nrn/bin/nrnivmodl"
        if base_process == alt_path:
            exit(-1)
        else:
            print("trying {}".format(alt_path))
            return nrnivmodl(tries_left=tries_left - 1, base_process=alt_path)
    finally:
        if clean_after:
            compiled_files = functools.reduce(lambda l1, l2: l1 + l2,
                                              [glob.glob("*.{}".format(file_type)) for file_type in ["o", "c"]])
            print("Deleting {}".format(compiled_files))
            for compiled_file in compiled_files:
                os.remove(compiled_file)

    return compiler_output, err


def main(path=None, dest=None, mod=None, hoc=None):
    if mod:
        mod_path = path
        hoc_path = None
    elif hoc:
        mod_path = None
        hoc_path = path
    elif path:
        # path is to lib
        mod_path = path + "/mod_files"
        hoc_path = path + "/hoc_files"
    else:
        mod_path = './'
        hoc_path = './'
    if mod_path is not None:
        with cd(mod_path):
            compiler_output, err = nrnivmodl(clean_after=True)

    print("COMPILER:\n", compiler_output)
    print("ERROR:\n", err)
    if dest is not None:
        print("DEST:", dest)
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            nrnmech = "x86_64"
        elif platform.system() == 'Windows':
            nrnmech = "nrnmech.dll"
        else:
            print("unknown system")
            sys.exit(-1)
        # copy mod files
        try:
            shutil.copy2(mod_path + "/" + nrnmech, dest)
        # eg. src and dest are the same file
        except shutil.Error as e:
            print('Error: %s' % e)
        # eg. source or destination doesn't exist
        except IOError as e:
            print('Error: %s' % e.strerror)
        # copy hoc files
        hoc_files = glob.glob(hoc_path + "/*.hoc")
        for hoc_file in hoc_files:
            short_path = os.path.basename(hoc_file)
            try:
                shutil.copy2(hoc_file, dest + "/" + short_path)
            # eg. src and dest are the same file
            except shutil.Error as e:
                print('Error: %s' % e)
            # eg. source or destination doesn't exist
            except IOError as e:
                print('Error: %s' % e.strerror)
    return compiler_output if err is None else err


if __name__ == "__main__":
    # command line program
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./lib",
                        help="path to directory or files")
    parser.add_argument("--dest", type=str, default=None,
                        help="output destination")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hoc", action="store_true",
                       help="path is to '.hoc' files only (tries to compile mod files from current directory)")
    group.add_argument("--mod", action="store_true",
                       help="path is to '.mod' files only")
    args = parser.parse_args()
    main(*args._get_args(), **args._get_kwargs())
