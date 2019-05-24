# Making QMNIST

This file explains how to regenerate the QMNIST files. 

### Prerequisites

The generation of the QWMNIST files is best performed
on a linux computer running Debian or Ubuntu.

* Install `pytorch`, version 1.0, following the instructions of the [pytorch web site](http://pytorch.org).  We recommend using the anaconda method.

* Install the python package [`lap`][(https://pypi.org/project/lap/). We recommend doing so by running the following command
inside the pytorch conda environment.
```
(base) host:~$ pip install lap
```

* Install and compile Lush1 from Subversion. If running on Debian or Ubuntu, make sure you have installed the packages `build-essential`, `automake`, `libx11-dev`, `libreadline-dev`, `binutils-dev`, `imagemagick`, `indent`, `gzip`.  If running on other variants of Linux, try to get the same things. Then 
```
(base) host:~$ svn checkout https://svn.code.sf.net/p/lush/code/lush1/trunk lush
(base) host:~$ cd lush
(base) host:~$ configure
(base) host:~$ make
```

* Download the file `1stEdition1995.zip` from the NIST Special Database 19 page at <https://www.nist.gov/srd/nist-special-database-19>. Unpack it in same directory as this `README.md` file.

* Setup a virtual machine running Debian Etch. We recommend using [VirtualBox](http://virtualbox.org). Make sure to install the packages `build-essential`, `automake`, `libpng12-dev` inside this machine.  


### Compiling and running the NIST software

Setup shared directories to make sure that your virtual machine can see the host filesystem containing the `making_qmnist` directory, which itself contains the `1stEdition1995` directory. Then run the following commands inside the virtual machine:

```
etch:~ cd /path/to/1stEdition1995/rnist-1.3.11
etch:rnist-1.3.11$ make distclean
etch:rnist-1.3.11$ patch -p1 < ../../rnist-1.3.11.patch
etch:rnist-1.3.11$ autoreconf
etch:rnist-1.3.11$ configure
etch:rnist-1.3.11$ make
```

The program of interest is called `rnist-1.3.11/bin/mistopng`. It may even run in the host machine if you're running Linux. Otherwise you'll need to run it inside the virtual machine again. Anyway, the purpose is to convert all the relevant `.mis` files into lots of `.png` files.

```
etch:~$ cd /path/to/making_qmnist/1stEdition1995
etch:1stEdition1995$ for n in ./data/by_write/hsf_*/f*/d*.mis ; do 
etch:1stEdition1995$   ./rnist-1.3.11/bin/mistopng $n 
etch:1stEdition1995$ done
```

At this point there should be lots of png files that you can see with

```
etch: 1stEdition1995 $ find data/by_write -name '*.png' -print
```

You can now close the virtual machine.

### Processing NIST digits

The next step consists in creating files containing all the NIST digits and all the labels in the QMNIST format. The QMNIST data will then be extracted from these files using the MNIST recipe.

```
(base) host:~$ cd /path/to/making_qmnist
(base) host:making_qmnist$ path/to/lush/bin/lush
LUSH Lisp Universal Shell (compiled on ...)
.....
? (load "xnist.lsh")
.....
? (do-xnist)
```
The creates large files named `xnist-images-idx3-ubyte` and `xnist-labels-idx2-int` containing preprocessed version of all the nist digits.
The next step is to assemble and reorder the QMNIST files.


### Assembling the QMNIST files

Still inside the Lush interpreter, type

```
? (do-qmnist)
```
This call creates a set of files named `qmnist-{train,test}-images-idx3-ubyte` and `qmnist-{train,test}-labels-idx2-int`. These files are not yet the final QMNIST files because they haven't been reordered to match the MNIST files. This is achieve with the following command inside the running Lush interpreter
```
? (do-qmnist-reordered)
```
This call first runs python programs that compare and match all the QMNIST and MNIST digits.  The resulting matches are 
then used to regenerate the files `qmnist-{train,test}-images-idx3-ubyte` and `qmnist-{train,test}-labels-idx2-int` in their proper order.
The resulting data files should match exactly those that come
with the QMNIST distribution. You can now exit the Lush interpreter
by typing Control+D

```
? <Ctrl+D>
Really quit [y/N] ?y
Bye!
```

### Running the LENET5 validation experiments

Copy the Lush demonstration file `packages/gblearn2/demos/lenet5.lsh` into the current directory and apply the patch

```
$ cp path/to/lush/packages/gblearn2/demos/lenet5.lsh .
$ patch < lenet5.lsh.patch
```

Then start Lush again and type:

```
(base) host:making_qmnist$ path/to/lush/bin/lush
LUSH Lisp Universal Shell (compiled on ...)
.....
? (load "lenet5.lsh")
.....
? ;; to train on mnist
? (do-mnist)
? ;; to train on qmnist
? (do-qmnist)
```

Running this using the genuine MNIST and QMNIST files should
give the results reported in the QMNIST `README.md` file. 
