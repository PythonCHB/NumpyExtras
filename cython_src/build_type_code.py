#!/usr/bin/env python

"""
simple templating script to build type code for cy_accumulator)

usage:

build_type_code path_to_numpy.pxd

"""


## for some reason not understood by me, I get a bunch of errors like:
## cython_src/cy_accumulator.c:1451: error: 'NPY_INT128' undeclared (first use in this function)
## even though I'm pulling these directly from the pxd file -- weird.
## some are there with the NPY_XXXXX enum, but don't have typedefs
## so this is a hack to keep the ones that give me errors out
not_supported = ['NPY_INT128',
                 'NPY_INT256',
                 'NPY_UINT128',
                 'NPY_UINT256',
                 'NPY_FLOAT16',
                 'NPY_FLOAT80',
                 'NPY_FLOAT96',
                 'NPY_FLOAT128',
                 'NPY_FLOAT256',
                 'NPY_COMPLEX32',
                 'NPY_COMPLEX160',
                 'NPY_COMPLEX192',
                 'NPY_COMPLEX256',
                 'NPY_COMPLEX512',
                 ]

import sys

try:
    filename = sys.argv[1]
    
except IndexError:
    print " you need to pass in the path to the numpy.pxd file"
    sys.exit()

try:
    pxd = file(filename, 'r').readlines()
except IOError:
    print "file: %s does not appear to be ther"%filename
    sys.exit()

print "parsing pxd file:", filename

# look for the enum:
n = 0
for n in range( len(pxd) ):
    if "".join(pxd[n].strip().split()) == "cdefenumNPY_TYPES:":
        break
else:
    raise Exception("got to end of file without finding enum")
print "enum found at line: ", n
# load them up
NPY_TYPES = []
for n in range(n+1, len(pxd)):
    line = pxd[n].strip()
    if not line:
        continue
    if not pxd[n].startswith(" "*8) : ## looking for end of the indent
        break
    NPY_TYPES.append(pxd[n].strip())

#map the np type names to the ENUMS
type_map = []
for TYPE in NPY_TYPES:
    if TYPE in not_supported: # leave these out
        continue
    if not TYPE[-1].isdigit(): # we only want the ones with byte sizes
        continue
    if TYPE[4:].startswith("INT"):
        np_name = "int" + TYPE[7:]
    elif TYPE[4:].startswith("UINT"):
        np_name = "uint" + TYPE[8:]
    elif TYPE[4:].startswith("FLOAT"):
        np_name = "float" + TYPE[9:]
    elif TYPE[4:].startswith("COMPLEX"):
        np_name = "complex" + TYPE[11:]
    type_map.append( (np_name, TYPE) )
#add one back in:
# type_map.append( ("void","NP_VOID") )

# build the code for the __init__ (setting the type flag)
#code = ["""        if self.dtype == np.uint8:
#            self.__type_case = cnp.NPY_UINT8"""
code = []
for type in type_map:
    code.append("""        elif self.dtype == np.%s:
            self.__type_case = cnp.%s"""%type )

# reolace the first "elif" with "if"
code[0] = code[0].replace("elif", "if")
code.append("""        else:
            raise NotImplementedError("This dtype is not supported")""" )


print "*********** __init__() code ************"
print "\n".join(code)

# build the code for append()

code = []
for type in type_map:
    code.append("""        elif self.__type_case == cnp.%s:
            (<cnp.%s_t*> self.__buffer)[self.length] = <cnp.%s_t> item"""%(type[1], type[0], type[0]) )

#        elif self.__type_case == cnp.NPY_INT32:
#            (<cnp.int32_t*> self.__buffer)[self.length] = <cnp.int32_t> item

# reolace the first "elif" with "if"
code[0] = code[0].replace("elif", "if")
code.append("""        else:
            raise NotImplementedError("This dtype is not supported for append")""" )

print "*********** append() code ************"
print "\n".join(code)


## build code for __get_item__

code = []
for type in type_map:
    code.append("""        elif self.__type_case == cnp.%s:
            item = (<cnp.%s_t*> self.__buffer)[index]"""%(type[1], type[0]) )
# replace the first "elif" with "if"
code[0] = code[0].replace("elif", "if")
code.append("""        else:
            raise NotImplementedError("This dtype is not supported for indexing")""" )

print "*********** __get_item__() code ************"

print "\n".join(code)
