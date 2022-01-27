""""
authors: Jakub Baran, Paulina Miśkowiec, Tomasz Borowski
"""
import math
import re
import numpy as np
from copy import deepcopy

# Class section
from typing import Tuple, List


class atom:
    def __init__(self, index, element, at_charge, coords, at_type=None, frozen=None, oniom_layer='L'):
        self.coords = coords
        self.index = index
        self.new_index = None
        self.element = element  # nazwa atomu
        self.at_type = at_type
        self.new_at_type = None
        self.at_charge = at_charge
        self.oniom_layer = oniom_layer
        self.tree_chain_classification = ''  #
        self.connect_list = []
        self.in_mainchain = False  #
        self.frozen = frozen  # 0 - not frozen, -1 - frozen
        self.name = None  #
        self.LAH = False  # True if this atom is a Link Atom Host (LAH)

    def __str__(self):
        return str(self.index) + " " + str(self.element) + " " + str(self.at_type) + " " + str(self.at_charge) + " " \
               + str(self.frozen) + " " + str(self.coords) + " " + str(self.oniom_layer)

    def get_coords(self):
        return self.coords

    def get_index(self):
        return self.index

    def get_new_index(self):
        return self.new_index

    def get_element(self):
        return self.element

    def get_type(self):
        return self.at_type

    def get_new_type(self):
        return self.new_at_type

    def get_at_charge(self):
        return self.at_charge

    def get_oniom_layer(self):
        return self.oniom_layer

    def get_tree_chain_classification(self):
        return self.tree_chain_classification

    def get_connect_list(self):
        return self.connect_list

    def get_in_mainchain(self):
        return self.in_mainchain

    def get_frozen(self):
        return self.frozen

    def get_name(self):
        return self.name

    def get_LAH(self):
        return self.LAH

    def set_coords(self, coords):
        self.coords = coords

    def set_type(self, at_type):
        self.at_type = at_type

    def set_new_type(self, new_at_type):
        self.new_at_type = new_at_type

    def set_at_charge(self, at_charge):
        self.at_charge = at_charge

    def set_oniom_layer(self, layer):
        self.oniom_layer = layer

    def set_tree_chain_classification(self, tree_chain_classification):
        self.tree_chain_classification = tree_chain_classification

    def set_connect_list(self, connect_list):
        self.connect_list = connect_list

    def set_in_mainchain(self, in_mainchain):
        self.in_mainchain = in_mainchain

    def set_frozen(self, frozen):
        self.frozen = frozen

    def set_new_index(self, new_index):
        self.new_index = new_index

    def set_name(self, name):
        self.name = name

    def set_LAH(self, LAH):
        self.LAH = LAH


class link_atom(atom):
    def __init__(self, coords, index, element,
                 at_type='', at_charge=0.0, bonded_to=None):
        atom.__init__(self, index, element, at_charge, coords, at_type=None, frozen=None, oniom_layer='L')
        self.at_type = at_type
        self.bonded_to = bonded_to
        self.H_coords = []
        self.mm_element = ''

    def __str__(self):
        return str(self.index) + " " + str(self.element) + " " + str(self.at_type) + " " + str(self.at_charge) + " " \
               + " " + str(self.H_coords) + " " + str(self.bonded_to)

    def get_type(self):
        return self.at_type

    def get_bonded_to(self):
        return self.bonded_to

    def get_MM_coords(self):
        return self.coords

    def get_H_coords(self):
        return self.H_coords

    def get_coords(self):
        return self.H_coords

    def get_mm_element(self):
        return self.mm_element

    def set_type(self, at_type):
        self.at_type = at_type

    def set_bonded_to(self, bonded_to):
        self.bonded_to = bonded_to

    def set_H_coords(self, H_coords):
        self.H_coords = H_coords

    def set_mm_element(self, mm_element):
        self.mm_element = mm_element


# Functions section

# Search in file

# def find_in_file() -> None:
#     """
#     create a dictionary with pointers to specific sections in the input file in order to have a easy access to them
#     :return:
#     Nothing because offset_dict is global and used in other functions
#     """
#     empty_line = 0  # count empty line
#     file.seek(0, 2)  # jump to the end of file
#     EOF = file.tell()  # set end of file (EOF) location
#     file.seek(0)  # jump to the beginning of the file
#     is_redundant_exists = False
#     while file.tell() != EOF:
#         previousLine = file.tell()
#         line = file.readline()

#         if empty_line == 1:
#             offsets["comment"] = previousLine
#             empty_line += 1  # avoid going again to this if statement
#         elif empty_line == 3:
#             offsets["chargeAndSpin"] = previousLine
#             offsets["atomInfo"] = file.tell()
#             empty_line += 1
#         elif empty_line == 5:
#             offsets["connectList"] = previousLine
#             empty_line += 1
#         elif empty_line == 7:
#             if re.match(r'^[0-9]+', line):
#                 offsets["redundant"] = previousLine
#                 is_redundant_exists = True
#             else:
#                 offsets["parm"] = previousLine
#             empty_line += 1
#         elif empty_line == 9 and is_redundant_exists:
#             offsets["parm"] = previousLine
#             empty_line += 1

#         if line == '\n':
#             empty_line += 1


def find_in_file(file):
    """
    create a dictionary with pointers to specific sections in the input file in order to have a easy access to them
    :input: file - a file object
    :returns: offsets - a dictionary
    """
    offsets = {"chargeAndSpin": -1,
           "atomInfo": -1,
           "comment": -1,
           "connectList": -1,
           "header": 0,
           "parm": -1,
           "redundant": -1
           }   
    
    empty_line = 0  # count empty line
    file.seek(0, 2)  # jump to the end of file
    EOF = file.tell()  # set end of file (EOF) location
    file.seek(0)  # jump to the beginning of the file
    is_redundant_exists = False
    while file.tell() != EOF:
        previousLine = file.tell()
        line = file.readline()

        if empty_line == 1:
            offsets["comment"] = previousLine
            empty_line += 1  # avoid going again to this if statement
        elif empty_line == 3:
            offsets["chargeAndSpin"] = previousLine
            offsets["atomInfo"] = file.tell()
            empty_line += 1
        elif empty_line == 5:
            offsets["connectList"] = previousLine
            empty_line += 1
        elif empty_line == 7:
            if re.match(r'^[0-9]+', line):
                offsets["redundant"] = previousLine
                is_redundant_exists = True
            else:
                offsets["parm"] = previousLine
            empty_line += 1
        elif empty_line == 9 and is_redundant_exists:
            offsets["parm"] = previousLine
            empty_line += 1

        if line == '\n':
            empty_line += 1

    file.seek(0)
    return offsets


# Read

def read_Chk(header):
    """
    find Chk file name and Chk extension in input file
    : header - a string with header from ONIOM input
    :return:
    list where [0] is file name and [1] is file extension
    """

    try:
        find_chk = re.search(r'%[Cc][Hh][Kk]=(.+)(\.[Cc][Hh][Kk]\s)', header)
        chk_core = find_chk.group(1)
        chk_extension = find_chk.group(2)
    except AttributeError:
        chk_core = ''
        chk_extension = '\n'  # if not found always go to next line

    return chk_core, chk_extension


# def read_Chk():
#     """
#     find Chk file name and Chk extension in input file
#     :return:
#     list where [0] is file name and [1] is file extension
#     """
#     header = read_header()

#     try:
#         find_chk = re.search(r'%[Cc][Hh][Kk]=(.+)(\.[Cc][Hh][Kk]\s)', header)
#         chk_core = find_chk.group(1)
#         chk_extension = find_chk.group(2)
#     except AttributeError:
#         chk_core = ''
#         chk_extension = '\n'  # if not found always go to next line

#     return chk_core, chk_extension


# def read_header() -> str:
#     """
#     save a file header section to string variable
#     :return:
#     str which contains the content of the header
#     """
#     file.seek(offsets["header"])
#     header_line = ""
#     while True:
#         line = file.readline()
#         if line == '\n':
#             break

#         header_line += line

#     return header_line

# def read_header(file, offset) -> str:
#     """
#     save a file header section to string variable
#     :input: file - file object
#     :returns:
#     str which contains the content of the header
#     """
#     header_line = ""
#     while True:
#         line = file.readline()
#         if line == '\n':
#             break

#         header_line += line

#     return header_line


# def read_comment() -> str:
#     """
#     save a file comment section to string variable
#     :return:
#     str which contains the content of the comment
#     """
#     file.seek(offsets["comment"])
#     comment_line = ""

#     while True:
#         line = file.readline()
#         if line == '\n':
#             break

#         comment_line += line

#     return comment_line

def read_from_to_empty_line(file, offset) -> str:
    """
    save a file fragment between the place specified by offset to an empty line
    to a string, which is returned
    :input: file - file object
    :returns:
    str which contains the content of the header
    """
    file.seek(offset)
    header_line = ""
    while True:
        line = file.readline()
        if line == '\n':
            break

        header_line += line

    return header_line


# def read_charge_spin() -> dict:
#     """
#     create a dictionary with information about charge and spin
#     :return:
#     dict which contains the content of the charge and spin
#     """
#     file.seek(offsets["chargeAndSpin"])
#     chargeAndSpin_list = file.readline().split()
#     chargeAndSpin_dict = {"ChrgRealLow": 0,
#                           "SpinRealLow": 0,
#                           "ChrgModelHigh": 0,
#                           "SpinModelHigh": 0,
#                           "ChrgModelLow": 0,
#                           "SpinModelLow": 0
#                           }
#     for count, key in enumerate(chargeAndSpin_dict):
#         chargeAndSpin_dict[key] = int(chargeAndSpin_list[count])

#     return chargeAndSpin_dict

def read_charge_spin(file, offset) -> dict:
    """
    read from file information about charge and spin
    :return:
    dict which contains the content of the charge and spin section of oniom input file
    """
#    2-layers ONIOM
#    chrgreal-low  spinreal-low  chrgmodel-high  spinmodel-high  chrgmodel-low  spinmodel-low

#    3-layer ONIOM
#    cRealL  sRealL   cIntM   sIntM   cIntL  sIntL   cModH  sModH   cModM  sModM   cModL   sModL
    two_layer_seq = ["ChrgRealLow", "SpinRealLow", "ChrgModelHigh", "SpinModelHigh",
                     "ChrgModelLow", "SpinModelLow"]
    
    three_layer_seq = ["ChrgRealLow", "SpinRealLow", "ChrgIntMed", "SpinIntMed",
                       "ChrgIntLow", "SpinIntLow", "ChrgModelHigh", "SpinModelHigh",
                       "ChrgModelMed", "SpinModelMed", "ChrgModelLow", "SpinModelLow"]

    file.seek(offset)
    chargeAndSpin_dict = {"ChrgRealLow": 0, "SpinRealLow": 0,
                          "ChrgModelHigh": 0, "SpinModelHigh": 0,
                          "ChrgModelMed": 0, "SpinModelMed": 0,
                          "ChrgModelLow": 0, "SpinModelLow": 0,
                          "ChrgIntMed": 0, "SpinIntMed": 0,
                          "ChrgIntLow": 0, "SpinIntLow": 0,}
    chargeAndSpin_list = file.readline().split()
    n_items = len(chargeAndSpin_list)
    if n_items == 6:
        for key, value in zip(two_layer_seq, chargeAndSpin_list):
            chargeAndSpin_dict[key] = int(value)
    elif n_items == 12:
        for key, value in zip(three_layer_seq, chargeAndSpin_list):
            chargeAndSpin_dict[key] = int(value)

    return chargeAndSpin_dict




def read_atom_inf(file, offset) -> tuple:
#def read_atom_inf() -> tuple:
    """
    create two list\n
    *atomObject_list:\n
    [0] - atom name\n
    [1] - atom type\n
    [2] - atom charge\n
    [3] - atom frozen\n
    [4] - atom x coords\n
    [5] - atom y coords\n
    [6] - atom z coords\n
    [7] - atom oniom_layer\n
    *linkObject_list:\n
    [0] - link atom name\n
    [1] - link atom type\n
    [2] - link atom charge\n
    [3] - link atom bonded_to\n
    :return:
    list where [0] is atom object and [1] is atom link object
    """
#    file.seek(offsets["atomInfo"])
    file.seek(offset)
    atomObject_list = []
    linkObject_list = []
    oniom_layer_H_and_LAH_index_list = []
    at_index = 0
    while True:
        line = file.readline()
        if line == '\n':
            break

        if check_if_line_correct(line):
            at_index += 1
            line = filter_line(line)

            if len(line[3]) > 2:  # if line[3] is frozen parameter, string has max 2 characters
                line.insert(3, None)

            element = line[0]
            at_type = line[1]
            at_charge = float(line[2])
            if line[3] is None:
                frozen = line[3]
            else:
                frozen = int(line[3])
            coords = [float(line[4]), float(line[5]), float(line[6])]
            onion_layer = line[7]
            if onion_layer == 'H':
                oniom_layer_H_and_LAH_index_list.append(at_index)

            atomObject = atom(index=at_index, element=element, at_type=at_type, at_charge=at_charge,
                              frozen=frozen, coords=coords, oniom_layer=onion_layer)

            if len(line) > 8:  # atom has link section
                atomObject.set_LAH(True)
                oniom_layer_H_and_LAH_index_list.append(at_index)
                link = link_atom(index=at_index, element=line[-4], at_type=line[-3], at_charge=float(line[-2]),
                                 coords=coords)
                bonded_to = int(line[-1])
                link.set_bonded_to(bonded_to)
                linkObject_list.append(link)

            atomObject_list.append(atomObject)

    for H_lk_atm in linkObject_list:  # calculate the link atoms coords
        qm_atom_index = H_lk_atm.get_bonded_to()
        adjust_HLA_coords(H_lk_atm, atomObject_list[qm_atom_index - 1])

    return atomObject_list, linkObject_list, oniom_layer_H_and_LAH_index_list  # [0] is atomObject_list,
    # [1] is linkObject_list [2] is oniom_layer_H_and_LAH_index_list when return


#def read_connect_list() -> dict:
def read_connect_list(file, offset) -> dict:    
    """
    create a dictionary where key is atom and value is a list of atoms(ints) connected to it
    :return:
    dict = { atom(int 1-based) : [connected_atom_1, connected_atom_2, ...],}
    """
#    file.seek(offsets["connectList"])
    file.seek(offset)
    connect_dict = {}

    while True:
        line = file.readline().split()
        if not line:
            break
        line = [int(float(elem)) for elem in line]  # convert from str to int - int(float(i)) because int(i) cause
        # problem when has to convert 1.0

        if line[0] not in connect_dict:
            connect_dict[line[0]] = []

        for i in line[1::2]:
            connect_dict[line[0]].append(i)
            if i not in connect_dict:
                connect_dict[i] = []
            connect_dict[i].append(line[0])

    return connect_dict


# def read_parm() -> str:
#     """
#     save a file parm section to string variable
#     :return:
#     str which contains the content of the parm
#     """
#     file.seek(offsets["parm"])
#     parm_line = ""

#     while True:
#         line = file.readline()
#         if line == '\n':
#             break

#         parm_line += line

#     return parm_line


# def read_redundant() -> str:
#     """
#     save a file redundant section to string variable
#     :return:
#     str which contains the content of the redundant
#     """
#     file.seek(offsets["redundant"])
#     redundant_line = ""

#     while True:
#         line = file.readline()
#         if line == '\n':
#             break

#         redundant_line += line

#     return redundant_line


# Auxiliary function

def filter_line(line: str) -> list:
    """
    make from line(str) a list that contains only necessary information about atoms\n
    example:\n
    line = N-N3--0.1592    		13.413952     49.941155     40.112556	L\n
    after exec of this function\n
    line = ['N', 'N3', '0.1592', '13.413952', '49.941155', '40.112556', 'L']\n
    :param line: line from file that contains information about atoms:
    :return: a list where every list[index] contains a different information about ato
    """
    for i in range(1, len(line)):
        if line[i] == '-' and not re.match(r'\s+', line[i - 1]):
            line = line[:i] + " " + line[i + 1:]

    filterPattern = re.compile(r'\s+')
    line = filterPattern.split(line)
    line.remove('')
    return line


def check_if_line_correct(line: str) -> bool:
    """
    check if read line contains correct format information about atom\n
    :param line: line to check
    :return: True if line is correct False if line is incorrect
    """
    pattern = r"[A-Z](.*[0-9]+[.][0-9]+){3}"
    if re.match(pattern, line):
        return True
    else:
        return False


def adjust_HLA_coords(H_lk_atm: link_atom, qm_atom: atom, bl_sf=0.723886) -> None:
    """
    calculates and sets coordinates of the H link atom using the provided scaling factor
    H_lk_atm : Hydrogen link_atom
    qm_atom : qm atom to which the link atom is connected to
    bl_sf : float, bond length scaling factor
    """
    qm_at_coords = np.array(qm_atom.get_coords())
    link_at_coords = np.array(H_lk_atm.get_MM_coords())
    H_coords = list((1.0 - bl_sf) * qm_at_coords + bl_sf * link_at_coords)
    H_lk_atm.set_H_coords(H_coords)


# Write

def write_header(header) -> None:
    """
    print into a file the content of the header
    :param header: variable which contains header information
    :return:
    """
    file_output.write('%oldChk={}{}'.format(read_Chk()[0], read_Chk()[1]))
    header = re.sub(r'%[Cc][Hh][Kk]=(.+)(\.[Cc][Hh][Kk]\s)', '%Chk={}{}{}'.format(read_Chk()[0], '_new', read_Chk()[1]),
                    header)
    file_output.write(header)


def write_comment(comment) -> None:
    """
    print into a file the content of the comment
    :param comment: variable which contains comment information
    :return:
    """
    file_output.write('\n{}'.format(comment))


# def write_charge_spin(charge_spin) -> None:
#     """
#     print into a file the content of the charge_spin
#     :param charge_spin: variable which contains charge_spin information
#     :return:
#     """
#     file_output.write('\n')
#     for key in charge_spin:
#         file_output.write(str(charge_spin[key]) + "  ")
#     file_output.write('\n')

def write_charge_spin(charge_spin, nlayers):
    """
    print into a file the content of the charge_spin
    :param charge_spin: dictionary with charge and spin information
    nlayers - intiger, number of layers (2 or 3)
    :return: line - string
    """
    two_layer_seq = ["ChrgRealLow", "SpinRealLow", "ChrgModelHigh", "SpinModelHigh",
                     "ChrgModelLow", "SpinModelLow"]
    
    three_layer_seq = ["ChrgRealLow", "SpinRealLow", "ChrgIntMed", "SpinIntMed",
                       "ChrgIntLow", "SpinIntLow", "ChrgModelHigh", "SpinModelHigh",
                       "ChrgModelMed", "SpinModelMed", "ChrgModelLow", "SpinModelLow"]
    line = ''
    if nlayers == 2:
        for key in two_layer_seq:
            line = line + str(charge_spin[key]) + "  "
    elif nlayers == 3:
        for key in three_layer_seq:
            line = line + str(charge_spin[key]) + "  "        
    return line
                              
def charge_change(atom_inf: tuple, type: str) -> list:
    """
    change atom charge and create a list of modificated charges
    :param atom_inf:
    :param type:
    :return:
    """

    atomList = []
    connectList = read_connect_list()

    for atom in atom_inf[0]:
        layer = atom.get_oniom_layer()
        coords = atom.get_coords()
        at_charge = atom.get_at_charge()
        line = [coords[0], coords[1], coords[2], at_charge, layer]
        atomList.append(line)

    for atom in atom_inf[0]:
        if atom.get_oniom_layer() == 'L':
            for link in atom_inf[1]:
                if link.get_index() == atom.get_index():
                    link_coords = link.get_coords()
                    atomList[atom.get_index()-1][0] = link_coords[0]
                    atomList[atom.get_index()-1][1] = link_coords[1]
                    atomList[atom.get_index()-1][2] = link_coords[2]

                    if type == "z1":
                        atomList[atom.get_index() - 1][3] = 0

                    if type == "z2":
                        atomList[atom.get_index() - 1][3] = 0
                        connection = connectList[atom.get_index()]
                        print(connection)
                        for x in connection:
                            atomList[x-1][3] = 0

                    if type == "z3":
                        atomList[atom.get_index() - 1][3] = 0
                        connection = connectList[atom.get_index()]
                        print(connection)
                        for x in connection:
                            connection2 = connectList[x]
                            atomList[x - 1][3] = 0
                            for y in connection2:
                                atomList[y-1][3] = 0

                    if type == "RC":
                        coords = atom.get_coords()
                        connection = connectList[atom.get_index()]
                        q = atomList[atom.get_index() - 1][3]/(len(connection)-1)
                        atomList[atom.get_index() - 1][3] = 0
                        for x in connection:
                            if atomList[x][4] == "L":
                                new_coords = half_distance(coords[0], coords[1], coords[2],
                                                      atomList[x - 1][0], atomList[x - 1][1], atomList[x - 1][2])
                                line = [new_coords[0], new_coords[1], new_coords[2], q, "P"]
                                atomList.append(line)

                    if type == "RCD":
                        coords = atom.get_coords()
                        connection = connectList[atom.get_index()]
                        q = atomList[atom.get_index() - 1][3]/(len(connection)-1)
                        atomList[atom.get_index() - 1][3] = 0
                        for x in connection:
                            if atomList[x][4] == "L":
                                new_coords = half_distance(coords[0], coords[1], coords[2],
                                                      atomList[x - 1][0], atomList[x - 1][1], atomList[x - 1][2])
                                line = [new_coords[0], new_coords[1], new_coords[2], q, "L"]
                                atomList.append(line)
                                atomList[x][3] -= 2*q

                    if type == "CS":
                        coords = atom.get_coords()
                        connection = connectList[atom.get_index()]
                        q = atomList[atom.get_index() - 1][3] / (len(connection) - 1)
                        atomList[atom.get_index() - 1][3] = 0
                        for x in connection:
                            if atomList[x][4] == "L":
                                new_coords = distance(coords[0], coords[1], coords[2],
                                                      atomList[x - 1][0], atomList[x - 1][1], atomList[x - 1][2])
                                line = [new_coords[0], new_coords[1], new_coords[2], 5*q, "L"]
                                atomList.append(line)
                                line = [new_coords[3], new_coords[4], new_coords[5], -5*q, "L"]
                                atomList.append(line)
                                atomList[x][3] += q

    return atomList


def half_distance(x1: float, y1: float, z1: float, x2: float, y2:float, z2:float) -> list:
    """
    calculate coords in half distance between two atoms
    :param x1:
    :param y1:
    :param z1:
    :param x2:
    :param y2:
    :param z2:
    :return:
    """

    cor = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]

    return cor


def distance(x1: float, y1: float, z1: float, x2: float, y2:float, z2:float) -> list:
    """
    calculate coords of point charges (CS)
    :param x1:
    :param y1:
    :param z1:
    :param x2:
    :param y2:
    :param z2:
    :return:
    """

    a1 = x1-x2
    b1 = y1-y2
    c1 = z1-z2

    a2 = -a1
    b2 = -b1
    c2 = -c1
    len = math.sqrt(a1*a1+b1*b1+c1*c1)

    a1 = a1/len
    b1 = b1/len
    c1 = c1/len

    a2 = a2/len
    b2 = b2/len
    c2 = c2/len
    cor = [0.1*a1*len, 0.1*b1*len, 0.1*c1*len, 0.1*a2*len, 0.1*b2*len, 0.1*c2*len]

    return cor


def write_change(atom_list: list, atom_change: tuple) -> None:
    """
    write the modifications of charges and coords into atom_change
    :param atom_list:
    :param atom_change:
    :return:
    """

    for atom in atom_change[0]:
        if atom.get_at_charge() != atom_list[atom.get_index()-1][3]:
            atom.set_at_charge(atom_list[atom.get_index()-1][3])

        coords = atom.get_coords()
        if coords[0] != atom_list[atom.get_index()-1][0]:
            atom.set_coords([atom_list[atom.get_index()-1][0], atom_list[atom.get_index()-1][1], atom_list[atom.get_index()-1][2]])


def charge_summary(atom_inf: tuple, atom_list: list) -> None:
    """
    calculate sum of atoms' charges
    :param atom_inf:
    :param atom_list:
    :return:
    """

    before_H = 0
    before_all = 0
    before_L = 0
    after_H = 0
    after_all = 0
    after_L = 0
    after_point = 0
    points_exist = False
    max_index = 0

    for x in atom_inf[0]:
        max_index = x.get_index()
        before_all += x.get_at_charge()
        if x.get_oniom_layer() == "H":
            before_H += x.get_at_charge()
        if x.get_oniom_layer() == "L":
            before_L += x.get_at_charge()

    print("Charge sums before changes:")
    print("Charge sum of all = " + str(round(before_all, 6)))
    print("Charge sum - layer H = " + str(round(before_H, 6)))
    print("Charge sum - layer L = " + str(round(before_L, 6)))

    i = 1
    for x in atom_list:
        after_all += x[3]
        if i > max_index:
            points_exist = True
            after_point += x[3]
        else:
            if x[4] == "H":
                after_H += x[3]
            if x[4] == "L":
                after_L += x[3]
        i += 1

    print("Charge sums after changes:")
    print("Charge sum of all = " + str(round(after_all, 6)))
    print("Charge sum - layer H = " + str(round(after_H, 6)))
    print("Charge sum - layer L = " + str(round(after_L, 6)))
    if points_exist:
        print("Charge sum - point charges = " + str(round(after_point, 6)))
    else:
        print("Charge sum - point charges = there's no point charges.")

def make_xyz_line(element, coords):
    new_line = element + '\t\t' + '{:06.6f}'.format(coords[0]) + '     ' + \
        '{:06.6f}'.format(coords[1]) + '     ' + \
        '{:06.6f}'.format(coords[2])
    return new_line

def write_xyz_file(output_f, atoms_list, link_atom_list, layer):
    """
    Parameters
    ----------
    output_f : FILE
        file to which xyz content is written.
    atoms_list : LIST
        list of atom objects from which information on element and coordinates 
        and layer are taken.
    link_atom_list : LIST
        list of link atom objects
    layer : STRING
        "HML", "HM" or "H" - all, H- and M-layers or H-layer (plus link atoms)

    Returns
    -------
    None.

    """
    link_atoms = 0
    H_atoms = 0
    M_atoms = 0
    L_atoms = 0
    
    list_of_lines = []

    for atom in atoms_list:
        element = atom.get_element()
        coords = atom.get_coords()
        oniom_layer = atom.get_oniom_layer()        

        if oniom_layer == "H":
            H_atoms += 1
        elif oniom_layer == "L":
            L_atoms += 1
        elif oniom_layer == "M":
            M_atoms += 1

        if layer == "HML":
            new_line = make_xyz_line(element, coords)
            list_of_lines.append(new_line)

        elif layer == "HM":
            if (oniom_layer == "L"):
                if atom.get_LAH():
                    for lk_atom in link_atom_list:
                        if atom.get_index() == lk_atom.get_index():
                            H_coords = lk_atom.get_H_coords()
                    element = "H"
                    coords = H_coords
                    link_atoms += 1
                    new_line = make_xyz_line(element, coords)
                    list_of_lines.append(new_line)       
            elif (oniom_layer == "H") or (oniom_layer == "M"):
                new_line = make_xyz_line(element, coords)
                list_of_lines.append(new_line)

        elif layer == "H":
            if (oniom_layer == "L") or (oniom_layer == "M"):
                if atom.get_LAH():
                    for lk_atom in link_atom_list:
                        if atom.get_index() == lk_atom.get_index():
                            H_coords = lk_atom.get_H_coords()
                    element = "H"
                    coords = H_coords
                    link_atoms += 1
                    new_line = make_xyz_line(element, coords)
                    list_of_lines.append(new_line)
            elif (oniom_layer == "H"):
                new_line = make_xyz_line(element, coords)
                list_of_lines.append(new_line)                
    
    
    if layer == "HML":
        n_atoms = H_atoms + M_atoms + L_atoms
    elif layer == "HM":
        n_atoms = H_atoms + M_atoms + link_atoms
    elif layer == "H":
        n_atoms = H_atoms + link_atoms
    
    list_of_lines.insert(0, " ")
    list_of_lines.insert(0, str(n_atoms))
    for item in list_of_lines:
        output_f.write("%s\n" % item)



def write_new_file(atom_list: list, type: str, atom_inf: tuple) -> None:
    """
    create file with coords and charges after modifications
    :param atom_list:
    :param type:
    :param atom_inf:
    :return:
    """

    output = open("output_file", 'a')

    header = re.sub(r'%[Cc][Hh][Kk]=(.+)(\.[Cc][Hh][Kk]\s)', '%Chk={}{}{}'.format(read_Chk()[0], '_new', read_Chk()[1]),\
                    read_header())
    output.write(header)
    output.write('\n')
    output.write("Model: " + type)
    output.write('\n')
    output.write('\n')

    charge_spin = read_charge_spin()

    output.write(str(charge_spin["ChrgModelHigh"]) + "\t" + str(charge_spin["SpinModelHigh"]))
    output.write('\n')

    for atom in atom_inf[0]:
        element = atom.get_element()
        coords = atom.get_coords()
        oniom_layer = atom.get_oniom_layer()

        if oniom_layer == 'H':
            type = atom.get_type()
            output.write(element + '\t\t' + '{:06.6f}'.format(coords[0]) + '     ' + \
                         '{:06.6f}'.format(coords[1]) + '     ' + \
                         '{:06.6f}'.format(coords[2]))
            output.write('\n')
        if atom.get_LAH():
            for x in atom_inf[1]:
                if atom.get_index() == x.get_index():
                    H_coords = x.get_H_coords()
                    output.write('H' + '\t\t' + \
                                 '{:06.6f}'.format(H_coords[0]) + '     ' + \
                                 '{:06.6f}'.format(H_coords[1]) + '     ' + \
                                 '{:06.6f}'.format(H_coords[2]))
                    output.write('\n')

    output.write('\n')

    for x in atom_list:
        if x[4] == 'L':
            output.write('{:06.6f}'.format(x[0]) + '     ' + \
                         '{:06.6f}'.format(x[1]) + '     ' + \
                         '{:06.6f}'.format(x[2]) + '\t' + str(round(x[3], 6)))
            output.write('\n')

    output.close()


def write_atom_inf(atom_inf: tuple) -> None:
    """
    print into a file the content of the atom_inf
    :param atom_inf: variable which contains atom_inf information
    :return:
    """
    for atom in atom_inf[0]:
        element = atom.get_element()
        type = atom.get_type()
        at_charge = atom.get_at_charge()
        frozen = atom.get_frozen()
        if frozen is None:
            frozen = ''
        coords = atom.get_coords()
        oniom_layer = atom.get_oniom_layer()
        file_output.write(element + '-' + type + '-' + str(round(at_charge, 6)) + '\t' + str(frozen) + '\t\t' + \
                          '{:06.6f}'.format(coords[0]) + '     ' + \
                          '{:06.6f}'.format(coords[1]) + '     ' + \
                          '{:06.6f}'.format(coords[2]) + '\t' + oniom_layer)
        for link in atom_inf[1]:
            if link.get_index() == atom.get_index():
                link_element = link.get_element()
                link_type = link.at_type
                link_charge = link.get_at_charge()
                link_bonded_to = link.get_bonded_to()
                file_output.write('  ' + link_element + '-' + link_type + '-' + '{:02.6f}'.format(link_charge) + \
                                  '\t' + str(link_bonded_to))
        file_output.write('\n')


# def write_atom_inf_change(atom_inf: tuple, atom_list: list) -> int:
#     """
#     print into a file the content of the atom_inf_change
#     :param atom_inf: variable which contains atom_inf information
#     :return:
#     """
#     i=0
#     for atom in atom_inf[0]:
#         element = atom.get_element()
#         type = atom.get_type()
#         at_charge = atom.get_at_charge()
#         frozen = atom.get_frozen()
#         if frozen is None:
#             frozen = ''
#         coords = atom.get_coords()
#         oniom_layer = atom.get_oniom_layer()
#         file_output.write(element + '-' + type + '-' + str(round(at_charge, 6)) + '\t' + str(frozen) + '\t\t' + \
#                           '{:06.6f}'.format(coords[0]) + '     ' + \
#                           '{:06.6f}'.format(coords[1]) + '     ' + \
#                           '{:06.6f}'.format(coords[2]) + '\t' + oniom_layer)
#         for link in atom_inf[1]:
#             if link.get_index() == atom.get_index():
#                 link_element = link.get_element()
#                 link_type = link.at_type
#                 link_charge = link.get_at_charge()
#                 link_bonded_to = link.get_bonded_to()
#                 file_output.write('  ' + link_element + '-' + link_type + '-' + '{:02.6f}'.format(link_charge) + \
#                                   '\t' + str(link_bonded_to))
#         file_output.write('\n')
#         i += 1

#     return i


def write_oniom_atom_section(file, atom_list, link_atom_list):
    """
    print into a file (ONIOM input) the atom section 
    :param 
    file - file object (to write to)
    atom_list: list of atom objects
    link_atom_list: list of link atom objects        
    :return:
    """
    for atom in atom_list:
        element = atom.get_element()
        type = atom.get_type()
        at_charge = atom.get_at_charge()
        frozen = atom.get_frozen()
        if frozen is None:
            frozen = ''
        coords = atom.get_coords()
        oniom_layer = atom.get_oniom_layer()
        file.write(element + '-' + type + '-' + str(round(at_charge, 6)) + '\t' + str(frozen) + '\t\t' + \
                          '{:06.6f}'.format(coords[0]) + '     ' + \
                          '{:06.6f}'.format(coords[1]) + '     ' + \
                          '{:06.6f}'.format(coords[2]) + '\t' + oniom_layer)
        if atom.get_LAH:
            for link in link_atom_list:
                if link.get_index() == atom.get_index():
                    link_element = link.get_element()
                    link_type = link.at_type
                    link_charge = link.get_at_charge()
                    link_bonded_to = link.get_bonded_to()
                    file.write('  ' + link_element + '-' + link_type + '-' + '{:02.6f}'.format(link_charge) + \
                                  '\t' + str(link_bonded_to))
                    break
        file.write('\n')


def write_points_charges(i: int, atom_list: list) -> None:
    """
    print into file point charges
    :param i:
    :param atom_list:
    :return:
    """

    file_output.write('\n')
    for x in range (i, len(atom_list)):
        file_output.write('{:06.6f}'.format(atom_list[x][0]) + '     ' + \
                     '{:06.6f}'.format(atom_list[x][1]) + '     ' + \
                     '{:06.6f}'.format(atom_list[x][2]) + '\t' + str(round(atom_list[x][3], 6)))
        file_output.write('\n')


# def write_connect(connect) -> None:
#     """
#     print into a file the content of the connect
#     :param connect: variable which contains connect section information
#     :return:
#     """
#     file_output.write('\n')
#     for key, value in sorted(connect.items()):
#         value = [i for i in value if i > key]  # remove information about redundant connection
#         file_output.write(str(key) + " " + " ".join(str(item) + " 1.0" for item in value) + '\n')

def write_connect(file, connect) -> None:
    """
    print into a file the content of the connect
    :param connect - a dictionary with connectivity information
    file - file object (to write to)
    :return:
    """
    for key, value in sorted(connect.items()):
        value = [i for i in value if i > key]  # remove information about redundant connection
        file.write(" " + str(key) + " " + " ".join(str(item) + " 1.0" for item in value) + ' \n')


def write_redundant(redundant) -> None:
    """
    print into a file the content of the redundant
    :param redundant: variable which contains redundant section information
    :return:
    """
    file_output.write('\n{}'.format(redundant))


def write_parm(parm) -> None:
    """
    print into a file the content of the parm
    :param parm: variable which contains parm section information
    :return:
    """
    file_output.write('\n{}'.format(parm))


def read_xyz_file(file):
    """
    reads xyz file
    file - file object
    :returns: xyz_atoms - a list of lists, each [str - element symbol, float -x coord, float - y, float - z]
    """
    xyz_atoms = []
    file.seek(0)

    line = file.readline()  # in order to read number of atoms
    line = line.split()
    n_read = int(line[0]) # number of atoms read from xyz header
    
    file.readline() # skip comment line

    while True:
        line = file.readline()
        if line == '':
            break

        line = line.split()
        element = line[0]
        coords_x = float(line[1])
        coords_y = float(line[2])
        coords_z = float(line[3])
        xyz_atoms.append([element, coords_x, coords_y, coords_z])

    if n_read != len(xyz_atoms):
        print("in function read_xyz_file: number of atoms read from file header: ", n_read, \
              " does not match number of atoms found in the coordinate section: ", len(xyz_atoms))
    return xyz_atoms


def new_coords(fileName_xyz: str, atomObject_list: Tuple[List[atom], List[link_atom], List[int]]):
    """
    change the coords values of all atoms
    :param fileName_xyz: file that contains new atoms coords
    :param atomObject_list: list of atoms which coords we want to change (use list which atom_inf function return)
    :return:
    """
    file_xyz = open("{}".format(fileName_xyz), 'r')

    file_xyz.readline()  # in order to jump to coords section
    file_xyz.readline()

    ctr = 0

    while True:
        line = file_xyz.readline()
        if line == '':
            break

        line = line.split()
        coords_x = float(line[1])
        coords_y = float(line[2])
        coords_z = float(line[3])
        atomObject_list[0][ctr].set_coords([coords_x, coords_y, coords_z])
        ctr += 1

    file_xyz.close()


def new_charge(fileName_qout: str, atomObject_list):
    """
    change the charge values of atoms which oniom_layer set to H
    :param fileName_qout: file that contains new atoms charge
    :param atomObject_list: list of atoms which coords we want to change (use list which atom_inf function return)
    :return:
    """
    file_qout = open("{}".format(fileName_qout), 'r')
    new_charge_list = file_qout.read().split()
    new_charge_list = [float(i) for i in new_charge_list]
    ctr = 0
    for index_atom in atomObject_list[2]:
        if atomObject_list[0][index_atom - 1].get_oniom_layer() == 'L':
            ctr += 1
            continue
        atomObject_list[0][index_atom - 1].set_at_charge(new_charge_list[ctr])
        ctr += 1

    file_qout.close()


# Main

if __name__ == '__main__':
    # Sample input
    file = open("h6h-oxo+succinate+water_hyo_17_07_b.gjf", 'r')
    offsets = {"chargeAndSpin": -1,
               "atomInfo": -1,
               "comment": -1,
               "connectList": -1,
               "header": 0,
               "parm": -1,
               "redundant": -1
               }
    find_in_file()

    # read information from file
    header = read_header()
    comment = read_comment()
    charge_and_spin = read_charge_spin()
    atom = read_atom_inf()
    atom_change = deepcopy(atom)

    connect = read_connect_list()
    parm = read_parm()


    # simple print on screen information
    # print(header)
    # print(comment)
    # print(charge_and_spin)
    # for i in range(0, len(atom[0])):
    #     print(atom[0][i])
    # for i in range(0, len(atom[1])):
    #     print(atom[1][i])
    # print(connect)
    # print(parm)

   # modify information - charge and coords
   # new_charge("h6h.qout", atom)
   # new_coords("h6h.xyz", atom)

    # print modify information

    # save information in output file
    file_output = open("output", 'a')


    write_header(header)
    write_comment(comment)
    write_charge_spin(charge_and_spin)
    write_atom_inf(atom)
    write_connect(connect)
    write_parm(parm)


    file_output.close()

    file_output = open("output_file", 'a')

    #write_header(header)
    # write_comment(comment)
    # write_charge_spin(charge_and_spin)
    # write_atom_inf_new(atom)

    atom_output = charge_change(atom, 'RCD')
    write_change(atom_output, atom_change)
    charge_summary(atom, atom_output)
    write_new_file(atom_output, 'CS', atom)


    file_output.close()

    file_output = open("output_change", 'a')
    write_header(header)
    write_comment(comment)
    write_charge_spin(charge_and_spin)
    ctr = write_atom_inf_change(atom_change, atom_output)
    write_connect(connect)
    write_parm(parm)
    write_points_charges(ctr, atom_output)
    file_output.close()


    file.close()
