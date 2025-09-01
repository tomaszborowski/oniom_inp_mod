""""
authors: Jakub Baran, Paulina Miśkowiec, Tomasz Borowski

last update: 1 September 2025
"""
import re, math, scipy, string, scipy.spatial
import numpy as np
from copy import deepcopy

# CONSTANTS
letters = string.ascii_uppercase
digits = string.digits

vdw_radii = {\
'Sc':1.648, 'Ti':1.588, 'V':1.572, 'Cr':1.512, 'Mn':1.481, 'Fe':1.456, 'Co':1.436, 'Ni':1.417, 'Cu':1.748, 'Zn':1.382,\
'Y':1.673, 'Zr':1.562, 'Nb':1.583, 'Mo':1.526,'Tc':1.499, 'Ru':1.482, 'Rh':1.465, 'Pd':1.450, 'Ag':1.574, 'Cd':1.424,\
'La':1.761,'Hf':1.571, 'Ta':1.585, 'W':1.535, 'Re':1.477, 'Os':1.560, 'Ir':1.420, 'Pt':1.377, 'Au':1.647, 'Hg':1.353} 
# values taken from the UFF force field: DOI: 10.1021/ja00051a040 (Table 1: nonbond distance/2.)


#######################
#### Class section ####
#######################
class point_charge:
    
    __slots__ = ["charge", "coords"]
    
    def __init__(self, charge, coords):
        self.charge = charge # float
        self.coords = coords # a list of 3 numbers
        
    def __str__(self):
        return  str(self.coords[0]) + " " + str(self.coords[1])  + " " +\
            str(self.coords[2]) + " " + str(self.charge)
    
    def get_coords(self):
        return self.coords

    def get_charge(self):
        return self.charge
    
    def get_string(self):
        x = '{:06.6f}'.format(self.coords[0])
        y = '{:06.6f}'.format(self.coords[1])
        z = '{:06.6f}'.format(self.coords[2])
        q = '{:06.8f}'.format(self.charge)
        if (self.coords[0] < 0.0):
            s1 = ''
        else:
            s1 = ' '
        if (self.coords[1] < 0.0):
            s2 = '  '
        else:
            s2 = '   '
        if (self.coords[2] < 0.0):
            s3 = '  '
        else:
            s3 = '   '            
        if (self.charge < 0.0):
            s4 = '  '
        else:
            s4 = '   '
        return  s1 + x + s2 + y + s3 + z + s4 + q
    
    def set_charge(self, q):
        self.charge = q
        
    def set_coords(self, crd):
        self.coords = crd


class atom:
    
    __slots__ = ["index", "element", "at_charge", "coords", "at_type", "frozen", "oniom_layer",
                 "new_index", "new_at_type", "tree_chain_classification", "connect_list",
                 "in_mainchain", "name", "LAH", "PDB_info"]
    
    def __init__(self, index, element, at_charge, coords, at_type=None, frozen=None, oniom_layer='L'):
        self.coords = coords
        self.index = index # 0-based (to be consistent with vmd and python indexing)
        self.new_index = None
        self.element = element  # element symbol
        self.at_type = at_type
        self.new_at_type = None
        self.at_charge = at_charge
        self.oniom_layer = oniom_layer # string, one from 'L', 'M' or 'H'
        self.tree_chain_classification = ''  #
        self.connect_list = []
        self.in_mainchain = False  #
        self.frozen = frozen  # 0 - not frozen, -1 - frozen
        self.name = None  #
        self.LAH = False  # True if this atom is a Link Atom Host (LAH)
        self.PDB_info = None  # PDB info about atom read from Gaussian input file (string)

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

    def get_PDB_info(self):
        return self.PDB_info

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

    def set_PDB_info(self, pdb_info):
        self.PDB_info = pdb_info

class link_atom(atom):
    
    __slots__ = ["index", "element", "at_charge", "coords", "at_type", "frozen", "oniom_layer",
                 "new_index", "new_at_type", "tree_chain_classification", "connect_list",
                 "in_mainchain", "name", "LAH", "bonded_to", "H_coords", "mm_element"]    
    
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


class residue:
    
    __slots__ = ["label", "index", "new_index", "in_protein", "atoms", "main_chain_atoms",
                 "trim", "chain"]
    
    def __init__(self, label, index):
        self.label = label
        self.index = index # zero based - to be consistent with vmd and python
        self.new_index = None
        self.in_protein = False
        self.atoms = [] # list of atom objects
        self.main_chain_atoms = [] # list of atom objects
        self.trim = False
        self.chain = ''

    def get_label(self):
        return self.label        
    def get_index(self):
        return self.index    
    def get_new_index(self):
        return self.new_index    
    def get_in_protein(self):
        return self.in_protein
    def get_atoms(self):
        return self.atoms
    def get_main_chain_atoms(self):
        return self.main_chain_atoms
    def get_side_chain_atoms(self):
        sc_atoms = self.atoms.copy()
        for at in self.main_chain_atoms:
            sc_atoms.remove(at)
        return sc_atoms   
    def next_atom(self):
        for atom in self.atoms:
            yield atom
    def get_trim(self):
        return self.trim
    def get_chain(self):
        return self.chain
    
    def set_in_protein(self,in_protein):
        self.in_protein = in_protein
    def add_atom(self,atom):
        self.atoms.append(atom)
        if atom.get_tree_chain_classification() == 'M':
            self.main_chain_atoms.append(atom)
    def add_main_chain_atom(self,atom):
            self.main_chain_atoms.append(atom)
    def set_trim(self,trim):
        self.trim = trim
    def set_new_index(self,new_index):
        self.new_index = new_index    
    def set_chain(self,chain_id):
        self.chain = chain_id

        
class peptide:
    
    __slots__ = ["label", "index", "is_peptide", "residues"]
    
    def __init__(self, label, index):
        self.label = label
        self.index = index  # 0-based     
        self.is_peptide = False
        self.residues = [] # list of residue objects

    def get_label(self):
        return self.label        
    def get_index(self):
        return self.index 
    def get_is_peptide(self):
        return self.is_peptide
    def get_residues(self):
        return self.residues
    def get_last_residue(self):
        if len(self.residues)>0:
            return self.residues[-1]
        else:
            return None
    def next_residue(self):
        for residue in self.residues:
            yield residue                         
    def add_residue(self,residue):
        self.residues.append(residue)   
        
        
#########################
####  Read functions ####
#########################
def find_in_file(file):
    """
    create a dictionary with pointers to specific sections in the input file 
    in order to have a easy access to them
    :input: file - a file object
    :returns: offsets - a dictionary
    """
    offsets = {"chargeAndSpin": -1,
           "atomInfo": -1,
           "comment": -1,
           "connectList": -1,
           "header": 0,
           "parm": -1,
           "redundant": -1,
           "p_charges": -1,
           "gen_basis": -1,
           "gen_ecp": -1
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
        # elif empty_line == 9:
        #     if re.match(r'^[0-9]+', line):
        #         offsets["p_charges"] = previousLine
        #     empty_line += 1
        # elif empty_line == 11 and is_redundant_exists:
        #     if re.match(r'^[0-9]+', line):
        #         offsets["p_charges"] = previousLine
        #     empty_line += 1
# UWAGA: to poniżej nie złapie wszystkich kombinacji w inpucie, trzebaby
# przepisać na inteligentniejsze skanowanie zawartości inputu
        elif empty_line == 9:
            if re.match(r'^[0-9]+', line):
                offsets["p_charges"] = previousLine
            elif re.match(r'^[a-zA-Z]+', line):
                offsets["gen_basis"] = previousLine
            empty_line += 1
        elif empty_line == 11 and is_redundant_exists:
            if re.match(r'^[0-9]+', line):
                offsets["p_charges"] = previousLine
            elif re.match(r'^[a-zA-Z]+', line):
                offsets["gen_basis"] = previousLine    
            empty_line += 1
        elif empty_line == 11:
            if re.match(r'^[a-zA-Z]+', line):
                offsets["gen_ecp"] = previousLine 
            empty_line += 1
        elif empty_line == 13 and is_redundant_exists:
            if re.match(r'^[a-zA-Z]+', line):
                offsets["gen_ecp"] = previousLine 
            empty_line += 1
            
        #if line == '\n':
        if line.replace(" ", "") == '\n':
            empty_line += 1

    file.seek(0)
    return offsets



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


def read_from_to_empty_line(file, offset) -> str:
    """
    save a file fragment between the place specified by offset to an empty line
    to a string, which is returned
    :input: file - file object
    offset - int, position within the file, as returned by file.tell()
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
    file.seek(offset)
    atomObject_list = []
    linkObject_list = []
    oniom_layer_H_and_LAH_index_list = []
    at_index = -1 # 0-based index
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
            #at_charge = float(line[2])
            f_charge = filter_charge(line[2])
            at_charge = f_charge[0]
            pdb_info = None
            if len(f_charge) > 1:
                pdb_info = f_charge[1]
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
            if pdb_info:
                atomObject.set_PDB_info(pdb_info)
            if len(line) > 8:  # atom has link section
                atomObject.set_LAH(True)
                oniom_layer_H_and_LAH_index_list.append(at_index)
                if line[-3][-1] == 'e' or line[-3][-1] == 'E':
                    charge_s = line[-3] + '-' + line[-2]
                    charge = float(charge_s)
                    atype = line[-4]
                    elem = line[-5]
                else:
                    charge = float(line[-2])
                    atype = line[-3]
                    elem = line[-4]
                #link = link_atom(index=at_index, element=line[-4], at_type=line[-3], at_charge=float(line[-2]),
                #                 coords=coords)
                link = link_atom(index=at_index, element=elem, at_type=atype, at_charge=charge,
                                 coords=coords)
                bonded_to = int(line[-1]) - 1 # read value is 1-based, internal values are 0-based
                link.set_bonded_to(bonded_to)
                linkObject_list.append(link)

            atomObject_list.append(atomObject)

    for H_lk_atm in linkObject_list:  # calculate the link atoms coords
        qm_atom_index = H_lk_atm.get_bonded_to()
        adjust_HLA_coords(H_lk_atm, atomObject_list[qm_atom_index]) # now 0-based index

    return atomObject_list, linkObject_list, oniom_layer_H_and_LAH_index_list  


def read_connect_list(file, offset) -> dict:    
    """
    create a dictionary where key is atom and value is a list of atoms(ints) connected to it
    keys and values are 0-based
    :return:
    dict = { atom(int 0-based) : [connected_atom_1, connected_atom_2, ...],}
    """
    file.seek(offset)
    connect_dict = {}

    while True:
        line = file.readline().split()
        if not line:
            break
        line = [int(float(elem)) for elem in line]  # convert from str to int - int(float(i)) because int(i) cause
        # problem when has to convert 1.0

        if (line[0]-1) not in connect_dict:
            connect_dict[(line[0]-1)] = []

        for i in line[1::2]:
            connect_dict[(line[0]-1)].append(i-1)
            if i-1 not in connect_dict:
                connect_dict[i-1] = []
            connect_dict[i-1].append((line[0]-1))

    return connect_dict


def read_p_charges(file, offset):
    """
    read point charge section from the Gaussian input file

    Parameters
    ----------
    file : file objects
        gaussian input file.
    offset : INT
        position in the file (as returned by .tell()) where
        the point charge section begins.

    Returns
    -------
    p_charges : LIST
        a list of point_charge objects.

    """
    file.seek(offset)
    p_charges = []
    
    while True:
        line = file.readline().split()
        if not line:
            break
        x = eval(line[0])
        y = eval(line[1])
        z = eval(line[2])
        coords = [x, y, z]
        charge = eval(line[3])
        
        p_ch = point_charge(charge, coords)
        p_charges.append(p_ch)
    
    return p_charges
    
    
def filter_line(line: str) -> list:
    """
    make from line(str) a list that contains only necessary information about atoms\n
    example:\n
    line = N-N3--0.1592    		13.413952     49.941155     40.112556	L\n
    after exec of this function\n
    line = ['N', 'N3', '-0.1592', '13.413952', '49.941155', '40.112556', 'L']\n
    :param line: line from file that contains information about atoms:
    :return: a list where every list[index] contains a different information about atom
    """
    for i in range(1, len(line)):
        if line[i] == '-' and not re.match(r'\s+', line[i - 1]):
            line = line[:i] + " " + line[i + 1:]

    filterPattern = re.compile(r'\s+')
    line = filterPattern.split(line)
    line.remove('')
    return line


def filter_charge(charge_frag: str) -> list:
    """
    get charge and optional PDB info from the string read from the input\n
    example:\n
    charge_frag = -0.011906(PDBName=CD,ResName=PRO,ResNum=426)
    charge_frag = -0.011906
    after execution this function returs:\n
    charge_list = [-0.011906, '(PDBName=CD,ResName=PRO,ResNum=426)']
    charge_list = [-0.011906]
    Parameters
    ----------
    charge_frag : str
        fragment of atom line containing info on atomic charge

    Returns
    -------
    list
        charge_list[0] - float, atomic charge
        optional charge_list[1] - string, PDB info
    """
    charge_list = charge_frag.split( '(' )
    charge_list[0] = eval(charge_list[0])
    if len(charge_list) > 1:
        charge_list[1] = '(' + charge_list[1]
    return charge_list


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


def read_qout_file(qout_f):
    """
    qout_f - file object
    :return: a list with atomic charges (floats)
    """
    new_charge_list = qout_f.read().split()
    new_charge_list = [float(i) for i in new_charge_list]
    return new_charge_list


def read_single_number(file, flag_line):
    """reads a single number from file and returns it as a numerical value 
    file : file object
    flag_line : string marker preceeding the value to be read"""
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:
            if len(a.split())>1:
                return eval(a.split()[1])
 
            
def read_single_string(file, flag_line):
    """reads a single string from file and returns it 
    file : file object
    flag_line : string marker preceeding the string to be read"""
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:
            if len(a.split())>1:
                return a.split()[1]


def read_rsi_index(file, flag_line, end_line):
    """reads residue sidechain and index lines 
    contained between lines starting with flag_line
    and end_line from file 
    ---
    file : file object
    flag_line : string
    end_line : string
    ---
    Returns lists:
    residue_index, sidechain_index and index_index
    """
    residue_index = []
    sidechain_index = []
    index_index = []
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            while True:
                a = file.readline()
                if not a:
                    break
                elif re.search(end_line,a):
                    break
                match_residue=re.search("residue",a)
                match_sidechain=re.search("sidechain",a)
                match_index=re.search("index",a)
                if match_residue:                            
                    temp = a.split()
                    temp.remove('residue')
                    for item in temp:
                        residue_index.append(eval(item))
                elif match_sidechain:
                    temp = a.split()
                    temp.remove('sidechain')
                    for item in temp:
                        sidechain_index.append(eval(item))
                elif match_index:
                    temp = a.split()
                    temp.remove('index')
                    for item in temp:
                        index_index.append(eval(item))
            for lista in [residue_index, sidechain_index, index_index]:
                temp = set(lista)
                temp_2 = list(temp)
                lista = temp_2.sort()
    return residue_index, sidechain_index, index_index


def read_rsi_names(file, flag_line, end_line):
    """reads residue names contained between lines starting with flag_line
    and end_line from file 
    ---
    file : file object
    flag_line : string
    end_line : string
    ---
    Returns a list:
    residue_names 
    """
    residue_names = []
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            while True:
                a = file.readline()
                if not a:
                    break
                elif re.search(end_line,a):
                    break
                match_residue=re.search("resname",a)
                if match_residue:                            
                    temp = a.split()
                    temp.remove('resname')
                    for item in temp:
                        residue_names.append(item)
    temp = set(residue_names)
    temp_2 = list(temp)
    temp_2.sort()
    return residue_names


def input_read_qm_part(file, residues, atoms, layer="H"):
    """
    Reads H_layer on M_layer from the input file
    Sets the oniom_layer attribute of atoms to 'H' or 'M' 
    Returns a list qm_part with atom objects making the QM part (H_layer or M_layer)
    (the list is sorted wrt atom index)
    ----
    file : file object
    residues : a list with residue obects for the system
    atoms : a list with atom obects for the system
    """   
    assert (type(residues) == list), "residues must be a list of residue objects"
    assert (type(atoms) == list), "atoms must be a list of atom objects"     
    qm_part = []
    if layer == "H":
        start_flag = "%H_layer"
        end_flag = "%end_H_layer"
    elif layer == "M":
        start_flag = "%M_layer"
        end_flag = "%end_M_layer"       
    residue_index,sidechain_index,index_index = read_rsi_index(file, start_flag, end_flag)
    if (len(residue_index) + len(sidechain_index) + len(index_index))>0:
        if layer == "H":
            print('\nThe H-layer of the system will consist of: ')
        elif layer == "M":
            print('\nThe M-layer of the system will consist of: ')
        print('residues: ',residue_index)
        print('and sidechains: ',sidechain_index)
        print('and atoms: ',index_index,'\n')
        for i in residue_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    qm_part.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in sidechain_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    if not at.get_in_mainchain():
                        qm_part.append(at)
            else:
                    print('\n Residue list does not have residue with index: ',i, '\n')
        for i in index_index:
            if atoms[i]:
                qm_part.append(atoms[i])
            else:
                print('\n Atom list does not have atom with index: ',i,'\n')
        temp = set(qm_part)
        qm_part = list(temp)
        qm_part.sort(key=lambda x: x.get_index(), reverse=True)
        for at in qm_part:
            if layer == "H":
                at.set_oniom_layer('H')
            elif layer == "M":
                at.set_oniom_layer('M')
    return qm_part 


def input_read_link_atoms(file, atoms, border="HL"):
    """
    Reads link_atoms section from the input file
    
    Returns a list link_atoms with link_atom objects 
    (the list is sorted wrt atom index)
    ----
    file : file object
    atoms : a list with atom obects for the system
    border : string, "HL", "HM" or "ML" - at which interface the link atoms are
    """   
    assert (type(atoms) == list), "atoms must be a list of atom objects"     
    MM_link_atoms = []
    HLA_types = []
    link_atoms = []
    index_index = []
    if border == "HL":
        flag_line = "%H_L_link_atoms"
        end_line = "%end_H_L_link_atoms"
        info_str = " HL "
        layer = "H"
    elif border == "HM":
        flag_line = "%H_M_link_atoms"
        end_line = "%end_H_M_link_atoms"
        info_str = " HM "
        layer = "H"
    elif border == "ML":
        flag_line = "%M_L_link_atoms"
        end_line = "%end_M_L_link_atoms"    
        info_str = " ML "
        layer = "M"
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            while True:
                a = file.readline()
                if not a:
                    break
                elif re.search(end_line,a):
                    break
                match_index=re.search("index",a)
                match_type=re.search("type",a)
                if match_index:
                    temp = a.split()
                    temp.remove('index')
                    for item in temp:
                        index_index.append(eval(item))
                elif match_type:
                    temp = a.split()
                    temp.remove('type')
                    for item in temp:
                        HLA_types.append(item)
            if len(index_index) == len(HLA_types) and len(index_index)>0 :
                index_index, HLA_types = (list(t) for t in zip(*sorted(zip(index_index, HLA_types))))
                print('\nThe atoms replaced by H-link atoms at the' + info_str + 'boarder are ')
                print('atoms with index: ', index_index)
                print('with H-link atom Amber atom types: ', HLA_types,'\n')
                for i in index_index:
                    if atoms[i]:
                        MM_link_atoms.append(atoms[i])
                    else:
                        print('Atom list does not have atom with index: ',i,'\n')
                for at,tp in zip(MM_link_atoms, HLA_types):
                    link_atoms.append(atom_to_link_atom(at, tp, 0.000001, layer))
    return link_atoms 


def input_read_freeze(file, residues, atoms):
    """
    Reads freeze data from the input file
    Sets the frozen atribute of atoms to be fixed to -1 
    Returns a list frozen with atom objects with frozen = -1
    (the list is sorted wrt atom index)
    ----
    file : file object
    residues : a list with residue obects for the system
    atoms : a list with atom objects for the system
    """   
    assert (type(residues) == list), "residues must be a list of residue objects"
    assert (type(atoms) == list), "atoms must be a list of atom objects"     
    frozen = []
    freeze_reference = []
    r_free = read_single_number(file,"%r_free")
    residue_index,sidechain_index,index_index = read_rsi_index(file, "%freeze_ref", "%end_freeze_ref")
    if (len(residue_index) + len(sidechain_index) + len(index_index))>0:
        print('\nThe reference with respect to which FREEZING will be done consist of: ')
        print('residues: ',residue_index)
        print('and sidechains: ',sidechain_index)
        print('and atoms: ',index_index)
        print('\nAll residues with at least one atom within: ',r_free,' A from the above reference will not be fixed \n')
        for i in residue_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    freeze_reference.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in sidechain_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    if not at.get_in_mainchain():
                        freeze_reference.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in index_index:
            if atoms[i]:
                freeze_reference.append(atoms[i])
            else:
                print('\n Atom list does not have atom with index: ',i, '\n')
        temp = set(freeze_reference)
        freeze_reference = list(temp)
        freeze_reference.sort(key=lambda x: x.get_index(), reverse=True)
            
        within_resids_bool = residues_within_r_from_atom_list(residues, freeze_reference, r_free)
        for w,resid in zip(within_resids_bool,residues):
            if not w:
                frozen.append(resid)
                for at in resid.get_atoms():
                    at.set_frozen(-1)
    return frozen 


def read_pdb_file(file):
    """
    reads content of the pdb file (in format as produced by prm2gaussian or oniom_inp_mod)
    returns a list of residue objects and a list of atom objects

    Parameters
    ----------
    file : file object
        file object with pdb file to be read.

    Returns
    -------
    residue_list : LIST
        a list of residue objects (their index runs from 0).
    atom_list : LIST
        a list of atom objects (their index runs from 0)
    """
    residue_list = []
    atom_list = []
    new_resid = None
    prev_resid_number = -1
    file.seek(0)
    while True:
        line = file.readline()
        if not line:
            if new_resid:
                if new_resid not in residue_list:
                    residue_list.append(new_resid)
            break
        line_split = line.split()
        if line_split[0] == 'ATOM':
            at_LAH = False
            at_number = eval(line_split[1])
            at_name = line_split[2]
            resid_name = line_split[3]
            resid_number = eval(line_split[4])
            x = eval(line_split[5])
            y = eval(line_split[6])
            z = eval(line_split[7])
            at_coord = [x, y, z]
            at_frozen = int(eval(line_split[8])) - 1 
            layer_info = int(eval(line_split[9]))
            if layer_info == 0:
                at_layer = 'L'
            elif layer_info == 1:
                at_layer = 'L'
                at_LAH = True
            elif layer_info == 2:
                at_layer = 'H'
            elif layer_info == 3:
                at_layer = 'M'
                at_LAH = True
            elif layer_info == 4:
                at_layer = 'M' 
            ele = line_split[10]
            
            at = atom(at_number - 1, ele, 0.0, at_coord, None, at_frozen, at_layer)
            at.set_name(at_name)
            at.set_LAH(at_LAH)
            
            atom_list.append(at)
            
            if resid_number != prev_resid_number:
                if new_resid:
                    residue_list.append(new_resid)
                new_resid = residue(resid_name, resid_number - 1)                    
                new_resid.add_atom(at)
                prev_resid_number = resid_number
            else:
                new_resid.add_atom(at)
                    
        elif line_split[0] == 'TER':
            pass
        elif line_split[0] == 'END':
            pass
        
    return residue_list, atom_list            

    
#########################
#### Write functions ####
#########################
def write_charge_spin(charge_spin, nlayers):
    """
    print into a file (Gaussian input) the content of the charge_spin
    :param charge_spin: dictionary with charge and spin information
    nlayers - intiger, number of layers (1, 2 or 3)
    :return: line - string
    """
    one_layer_seq = ["ChrgRealLow", "SpinRealLow"]

    two_layer_seq = ["ChrgRealLow", "SpinRealLow", "ChrgModelHigh", "SpinModelHigh",
                     "ChrgModelLow", "SpinModelLow"]
    
    three_layer_seq = ["ChrgRealLow", "SpinRealLow", "ChrgIntMed", "SpinIntMed",
                       "ChrgIntLow", "SpinIntLow", "ChrgModelHigh", "SpinModelHigh",
                       "ChrgModelMed", "SpinModelMed", "ChrgModelLow", "SpinModelLow"]
    line = ''
    if nlayers == 1:
        for key in one_layer_seq:
            line = line + str(charge_spin[key]) + "  "
    if nlayers == 2:
        for key in two_layer_seq:
            line = line + str(charge_spin[key]) + "  "
    elif nlayers == 3:
        for key in three_layer_seq:
            line = line + str(charge_spin[key]) + "  "        
    return line
                              

def make_xyz_line(element, coords):
    """
    generate an Element-coordinates line for Gaussian QM input file

    Parameters
    ----------
    element : STRING
        symbol of chemical element.
    coords : LIST
        a list of 3 numbers - coordinates of the atom.

    Returns
    -------
    new_line : STRING
        "El x y z \n" line.

    """
    new_line = element + '\t\t' + '{:06.6f}'.format(coords[0]) + '     ' + \
        '{:06.6f}'.format(coords[1]) + '     ' + \
        '{:06.6f}'.format(coords[2]) + "\n"
    return new_line


def write_xyz_file(output_f, atoms_list, link_atom_list, layer):
    """
    writes xyz file for a specified ONIOM (sub)system
    Parameters
    ----------
    output_f : FILE objects
        file to which xyz content is written.
    atoms_list : LIST
        list of atom objects from which information on element and coordinates 
        and layer are taken.
    link_atom_list : LIST
        list of link atom objects
    layer : STRING
        "HML", "HM" or "H" - all, H- and M-layers or H-layer (plus link atoms),
        which corresponds to "Real", "Intermediate" and "Model" ONIOM systems

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
                            if atoms_list[ lk_atom.get_bonded_to() ].get_oniom_layer() == "M":
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
                            if atoms_list[ lk_atom.get_bonded_to() ].get_oniom_layer() == "H":                            
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
    
    list_of_lines.insert(0, "\n")
    list_of_lines.insert(0, str(n_atoms)+"\n")
    for item in list_of_lines:
        output_f.write("%s" % item)


def gen_connect(atom_list):
    """
    generate a new connectivity list (0-based) for a list of atom objects

    Parameters
    ----------
    atom_list : LIST
        a list of atom objects.

    Returns
    -------
    connect : DICTIONARY
        a new connectivity dictionary: key - 0-based index, value - a list of indexes of connected atoms.

    """
    i = 0
    translate_index = {}
    for at in atom_list:
        at_ix = at.get_index()
        translate_index[at_ix] = i
        i += 1
    
    connect = {}
    for at in atom_list:
        at_ix = at.get_index()
        at_ix_new = translate_index[at_ix]
        at_connect = at.get_connect_list()
        at_connect_new = [translate_index[i] for i in at_connect]
        connect[at_ix_new] = at_connect_new
    
    return connect


def write_qm_input(file, header, comment, chargeSpin, atom_list, point_charge_list):
    """
    writes Gaussian QM input file

    Parameters
    ----------
    file : file object
        to which the content is written.
    header : string
        header of the input file.
    comment : string
        comment.
    chargeSpin : dictionary
        dic with charge and spin data.
    atom_list : list
        a list of atom objects.
    point_charge_list : list
        a list of point_charge objects.

    Returns
    -------
    None.

    """
#    connect = gen_connect(atom_list)
#    new_header = add_old_chk(header)
#    file.write(new_header)
    file.write(header)
    file.write('\n')
    file.write(comment)
    file.write('\n')


    file.write(str(chargeSpin["ChrgModelHigh"]) + "\t" + str(chargeSpin["SpinModelHigh"]))
    file.write('\n')

    for atom in atom_list:
        element = atom.get_element()
        coords = atom.get_coords()
        line = make_xyz_line(element, coords)
        file.write(line)
        
#    file.write('\n')
#    write_connect(file, connect)
    
    if len(point_charge_list) > 0:
        for p_q in point_charge_list:
            file.write( '\n{}'.format( p_q.get_string() ) )
        file.write('\n')
    
    file.write('\n')        


def write_mm_input(file, header, comment, chargeSpin, atom_list, point_charge_list, params=None):
    """
    writes Gaussian MM input file

    Parameters
    ----------
    file : file object
        to which the content is written.
    header : string
        header of the input file.
    comment : string
        comment.
    chargeSpin : dictionary
        dic with charge and spin data.
    atom_list : list
        a list of atom objects.
    point_charge_list : list
        a list of point_charge objects.
    params : string
        a string with parameters section
    Returns
    -------
    None.

    """
    connect = gen_connect(atom_list)
    new_header = add_old_chk(header)
    file.write(new_header)    
    file.write('\n')
    file.write(comment)
    file.write('\n')


    file.write(str(chargeSpin["ChrgModelLow"]) + "\t" + str(chargeSpin["SpinModelLow"]))
    file.write('\n')

    for atom in atom_list:
        element = atom.get_element()
        at_type = atom.get_type()
        at_charge = atom.get_at_charge()
        at_charge_str = str(round(at_charge, 6))
        at_pdb_info = atom.get_PDB_info()
        if at_pdb_info:
            at_charge_str += at_pdb_info
        coords = atom.get_coords()
        # line = element + '-' + at_type + '-' + str(round(at_charge, 6)) + '\t' + '\t\t' + \
        #                   '{:06.6f}'.format(coords[0]) + '     ' + \
        #                   '{:06.6f}'.format(coords[1]) + '     ' + \
        #                   '{:06.6f}'.format(coords[2]) + '\n'
        line = element + '-' + at_type + '-' + at_charge_str + '\t' + '\t\t' + \
                          '{:06.6f}'.format(coords[0]) + '     ' + \
                          '{:06.6f}'.format(coords[1]) + '     ' + \
                          '{:06.6f}'.format(coords[2]) + '\n'
        file.write(line)

    file.write('\n')
    write_connect(file, connect)

    if params:
        file.write('\n{}'.format(params))
    
    if len(point_charge_list) > 0:
        for p_q in point_charge_list:
            file.write( '\n{}'.format( p_q.get_string() ) )
        file.write('\n')
    
    file.write('\n') 

                
def write_oniom_atom_section(file, atom_list, link_atom_list):
    """
    print into a file (ONIOM input) the atom section 
    :param 
    file - file object (to write to)
    atom_list: list of atom objects
    link_atom_list: list of link atom objects        
    :return: None
    """
    for atom in atom_list:
        element = atom.get_element()
        at_type = atom.get_type()
        at_charge = atom.get_at_charge()
        at_charge_str = str(round(at_charge, 6))
        at_pdb_info = atom.get_PDB_info()
        if at_pdb_info:
            at_charge_str += at_pdb_info        
        frozen = atom.get_frozen()
        if frozen is None:
            frozen = ''
        coords = atom.get_coords()
        oniom_layer = atom.get_oniom_layer()
        # file.write(element + '-' + at_type + '-' + str(round(at_charge, 6)) + '\t' + str(frozen) + '\t\t' + \
        #                   '{:06.6f}'.format(coords[0]) + '     ' + \
        #                   '{:06.6f}'.format(coords[1]) + '     ' + \
        #                   '{:06.6f}'.format(coords[2]) + '\t' + oniom_layer)
        file.write(element + '-' + at_type + '-' + at_charge_str + '\t' + str(frozen) + '\t\t' + \
                          '{:06.6f}'.format(coords[0]) + '     ' + \
                          '{:06.6f}'.format(coords[1]) + '     ' + \
                          '{:06.6f}'.format(coords[2]) + '\t' + oniom_layer)            
        if atom.get_LAH:
            for link in link_atom_list:
                if link.get_index() == atom.get_index():
                    link_element = link.get_element()
                    link_type = link.at_type
                    link_charge = link.get_at_charge()
                    link_bonded_to = link.get_bonded_to() + 1 # because connec list is 0-based
                    file.write('  ' + link_element + '-' + link_type + '-' + '{:02.6f}'.format(link_charge) + \
                                  '\t' + str(link_bonded_to))
                    break
        file.write('\n')


def write_connect(file, connect) -> None:
    """
    print into a file the content of the connectivity list
    :param connect - a dictionary with connectivity information (0-based)
    file - file object (to write to)
    :return: None
    """
    for key, value in sorted(connect.items()):
        value = [i for i in value if i > key]  # remove information not needed in Gaussian connectivity list
        file.write(" " + str(key+1) + " " + " ".join(str(item+1) + " 1.0" for item in value) + ' \n')


def add_old_chk(header):
    chk_read = read_Chk(header)
    
    old_chk_line = '%oldChk={}{}'.format(chk_read[0], chk_read[1])
    new_header = re.sub(r'%[Cc][Hh][Kk]=(.+)(\.[Cc][Hh][Kk]\s)', '%Chk={}{}{}'.format(chk_read[0], '_new', chk_read[1]),
                    header)
    if len(chk_read[0]) == 0:
        return header
    else:
        new_header = old_chk_line + new_header
        return new_header

        
def write_oniom_inp_file(file, header, comment, charge_and_spin, nlayers,\
                         atoms_list, link_atoms_list, connect, redundant=None,\
                         params=None, p_charges=None, gen_basis=None, gen_ecp=None):
    """
    writes ONIOM input file

    Parameters
    ----------
    file : file object
        file to which ONIOM input will be written.
    header : STRING
        contains the header section of the input file.
    comment : STRING
        comment line(s) in the Gaussian input.
    charge_and_spin : DICTIONARY
        Dictionary with charge and multiplicity info.
    nlayers : INT
        number of layers: 2 or 3.
    atoms_list : LIST
        list of atom objects.
    link_atoms_list : LIST
        list of link atom objects.
    connect : DICTIONARY
        dictionary with atoms connectivity information.
    redundant : STRING, optional
        redundant coordinate section (which may follow connectivity). The default is None.
    params : STRING, optional
        section with FF parameters. The default is None.
    p_charges : LIST, optional
        a list of off-atom point charges (objects). The default is None.
    gen_basis : STRING, optional. 
        general basis set section. The default is None.
    gen_ecp : STRING, optional.
        general ECP section. The default is None.
    Returns
    -------
    None.

    """
    new_header = add_old_chk(header)
    file.write(new_header)
    file.write("\n")
    file.write(comment)
    file.write("\n")

#    2-layers ONIOM
#    chrgreal-low  spinreal-low  chrgmodel-high  spinmodel-high  chrgmodel-low  spinmodel-low
#
#    3-layer ONIOM
#    cRealL  sRealL   cIntM   sIntM   cIntL  sIntL   cModH  sModH   cModM  sModM   cModL   sModL
    
    charge_spin_line = write_charge_spin(charge_and_spin, nlayers)
    file.write(charge_spin_line + "\n")

    write_oniom_atom_section(file, atoms_list, link_atoms_list)
    file.write("\n")
    write_connect(file, connect)
    
    if redundant:
        file.write('\n{}'.format(redundant))
    
    if params:
        file.write('\n{}'.format(params))

    if p_charges:
        for p_q in p_charges:
            file.write('\n{}'.format( p_q.get_string() ) )
    
    if gen_basis:
        file.write("\n")
        file.write(gen_basis)
        
    if gen_ecp:
        file.write("\n")
        file.write(gen_ecp)        
    
    file.write("\n\n\n")     


# def write_oniom_inp_file(file, header, comment, charge_and_spin, nlayers,\
#                          atoms_list, link_atoms_list, connect, redundant=None, params=None, p_charges=None):
#     """
#     writes ONIOM input file

#     Parameters
#     ----------
#     file : file object
#         file to which ONIOM input will be written.
#     header : STRING
#         contains the header section of the input file.
#     comment : STRING
#         comment line(s) in the Gaussian input.
#     charge_and_spin : DICTIONARY
#         Dictionary with charge and multiplicity info.
#     nlayers : INT
#         number of layers: 2 or 3.
#     atoms_list : LIST
#         list of atom objects.
#     link_atoms_list : LIST
#         list of link atom objects.
#     connect : DICTIONARY
#         dictionary with atoms connectivity information.
#     redundant : STRING, optional
#         redundant coordinate section (which may follow connectivity). The default is None.
#     params : STRING, optional
#         section with FF parameters. The default is None.
#     p_charges : LIST, optional
#         a list of off-atom point charges (objects). The default is None.
#     Returns
#     -------
#     None.

#     """
#     new_header = add_old_chk(header)
#     file.write(new_header)
#     file.write("\n")
#     file.write(comment)
#     file.write("\n")

# #    2-layers ONIOM
# #    chrgreal-low  spinreal-low  chrgmodel-high  spinmodel-high  chrgmodel-low  spinmodel-low
# #
# #    3-layer ONIOM
# #    cRealL  sRealL   cIntM   sIntM   cIntL  sIntL   cModH  sModH   cModM  sModM   cModL   sModL
    
#     charge_spin_line = write_charge_spin(charge_and_spin, nlayers)
#     file.write(charge_spin_line + "\n")

#     write_oniom_atom_section(file, atoms_list, link_atoms_list)
#     file.write("\n")
#     write_connect(file, connect)
    
#     if redundant:
#         file.write('\n{}'.format(redundant))
    
#     if params:
#         file.write('\n{}'.format(params))

#     if p_charges:
#         for p_q in p_charges:
#             file.write('\n{}'.format( p_q.get_string() ) )
    
#     file.write("\n\n\n")


def write_mm_inp_file(file, header, comment, charge_and_spin,\
                         atoms_list, connect, redundant=None, params=None, p_charges=None):
    """
    writes MM input file

    Parameters
    ----------
    file : file object
        file to which ONIOM input will be written.
    header : STRING
        contains the header section of the input file.
    comment : STRING
        comment line(s) in the Gaussian input.
    charge_and_spin : DICTIONARY
        Dictionary with charge and multiplicity info.
    atoms_list : LIST
        list of atom objects.
    connect : DICTIONARY
        dictionary with atoms connectivity information.
    redundant : STRING, optional
        redundant coordinate section (which may follow connectivity). The default is None.
    params : STRING, optional
        section with FF parameters. The default is None.
    p_charges : LIST, optional
        a list of off-atom point charges (objects). The default is None.
    Returns
    -------
    None.

    """
    new_header = add_old_chk(header)
    file.write(new_header)
    file.write("\n")
    file.write(comment)
    file.write("\n")
    
    charge_spin_line = write_charge_spin(charge_and_spin, 1)
    file.write(charge_spin_line + "\n")

    mm_atoms_list = deepcopy(atoms_list)
    for at in mm_atoms_list:
        old_layer = at.get_oniom_layer()
        if old_layer != 'L':
            at.set_oniom_layer('L')
        if at.get_LAH:
            at.set_LAH(False)
        

    write_oniom_atom_section(file, mm_atoms_list, [])
    file.write("\n")
    write_connect(file, connect)
    
    if redundant:
        file.write('\n{}'.format(redundant))
    
    if params:
        file.write('\n{}'.format(params))

    if p_charges:
        for p_q in p_charges:
            file.write('\n{}'.format( p_q.get_string() ) )
    
    file.write("\n\n")   


def write_pdb_file(residue_list, file_name, write_Q=False):
    """writes a PDB file with a name file_name
    residue_list - a list of residue objects 
    only residues with trim=False are written to the file"""
    prev_resid_chain = ''
    pdb_file = open(file_name, 'w')
    residue_list_no_trim = []
    for residue in residue_list:
        if not residue.get_trim():
            residue_list_no_trim.append(residue)
    for residue in residue_list_no_trim:
        resid_name = residue.get_label()
        resid_name = resid_name.rjust(3, ' ')
        resid_number = str(residue.get_new_index() + 1)
        resid_number = resid_number.rjust(4, ' ')
        for atom in residue.get_atoms():
            ele = atom.get_element()
            ele = ele.rjust(2, ' ')
            at_coord = atom.get_coords()
            at_name = atom.get_name()
            at_name = at_name.ljust(4, ' ')
            at_number = atom.get_new_index() + 1
            at_number = str(at_number)
            at_number = at_number.rjust(5, ' ')
            at_layer = atom.get_oniom_layer()
            at_LAH = atom.get_LAH()
            if at_LAH and at_layer == 'L':
               at_beta = 1.0 
            elif at_layer == 'L': 
                at_beta = 0.0               
            elif at_layer == 'H':
                at_beta = 2.0
            elif at_LAH and at_layer == 'M':
                at_beta = 3.0
            elif at_layer == 'M':
                at_beta = 4.0                
            at_frozen = atom.get_frozen()    
            if at_frozen == 0:
                at_occupancy = 1.0
            else:
                at_occupancy = 0.0    
            at_charge = atom.get_at_charge()
            line = 'ATOM' + '  ' + at_number + ' ' +\
            at_name + ' ' + resid_name + '  ' + resid_number + '    ' +\
            '{:8.3f}'.format(at_coord[0]) + '{:8.3f}'.format(at_coord[1]) + '{:8.3f}'.format(at_coord[2]) +\
            '{:6.2f}'.format(at_occupancy) + '{:6.2f}'.format(at_beta) + '    ' + ele            
            if write_Q:
                line = line + ' ' + '{:5.3f}'.format(at_charge)
            line = line + '\n'
            pdb_file.write(line)
        if residue.get_new_index() == 0:
            prev_resid_chain = residue.get_chain()
            if residue_list_no_trim[1].get_chain() != prev_resid_chain:
                pdb_file.write('TER\n')
        elif residue.get_chain() != prev_resid_chain:
            prev_resid_chain = residue.get_chain()
            pdb_file.write('TER\n')
        else:
            pass
    pdb_file.close() 

        
#########################
#### Other functions ####
#########################
def NC_in_main_chain(residue):
    """ checks is the residue contains N(M) and C(M) atoms
    if yes, returns True, otherwise returns False"""
    temp = []
    for atom in residue.get_main_chain_atoms():
        temp.append(atom.get_element())
    main_chain_elements = set(temp)
    if 'N' in main_chain_elements and 'C' in main_chain_elements:
        return True
    else:
        return False


def is_peptide_bond2(residue_1,residue_2):
    """looks for O-C-N fragment between the two residues
    returns True if found, False otherwise
    assumes N is the first atom in a residue and CO are the last two in a residue
    in a peptide bond"""
    res1_atoms = residue_1.get_atoms()
    res2_atoms = residue_2.get_atoms()
    if len(res1_atoms) > 1 and len(res2_atoms) > 0:
        if res1_atoms[-2].get_element() == 'C' and res1_atoms[-1].get_element() == 'O'\
        and res2_atoms[0].get_element() == 'N'\
        and res1_atoms[-1].get_index() in res1_atoms[-2].get_connect_list()\
        and res2_atoms[0].get_index() in res1_atoms[-2].get_connect_list():
            return True
        else:
            return False
    else:
        return False 
    
    
def N_CO_in_residue(residue):
    """ checks is the residue contains N and CO as a first and two last atoms
    if yes, returns True, otherwise returns False"""
    temp = []
    for atom in residue.get_atoms():
        temp.append(atom.get_element())
    if len(temp) > 3:
        if temp[0] == 'N' and temp[-2] == 'C' and temp[-1] == 'O':
            return True
        elif temp[0] == 'N' and temp[-3] == 'C' and temp[-2] == 'O':
            return True
        else:
            return False
    else:
        return False
    
    
def main_side_chain(residue):
    """ determined which atoms are in the mainchain in a residue
    looks for N-C-C-O fragment and then H atoms bound to N or C
    and O atom bound to the second C 
    when found sets in_mainchain atribute of these atoms to True
    and adds them into main_chain_atoms list of this residue """
    for at1 in residue.get_atoms():
        if at1.get_element() == 'N':
            at1_connect = at1.get_connect_list()
            for at2 in residue.get_atoms():
                if at2.get_element() == 'C' and at2.get_index() in at1_connect:
                    at2_connect = at2.get_connect_list()
                    for at3 in residue.get_atoms():
                        if at3.get_element() == 'C' and at3.get_index() in at2_connect:
                            at3_connect = at3.get_connect_list()
                            for at4 in residue.get_atoms():
                                if at4.get_element() == 'O' and at4.get_index() in at3_connect:
                                    for at in [at1, at2, at3, at4]:
                                        at.set_in_mainchain(True)
                                        if at not in residue.get_main_chain_atoms():
                                            residue.add_main_chain_atom(at)
                                    for at5 in residue.get_atoms():
                                        if (at5.get_element() == 'H' and (at5.get_index() in at1_connect))\
                                        or (at5.get_element() == 'H' and (at5.get_index() in at2_connect))\
                                        or (at5.get_element() == 'O' and (at5.get_index() in at3_connect) and at5 != at4):
                                            at5.set_in_mainchain(True)
                                            residue.add_main_chain_atom(at5)


def atom_to_link_atom(at_ins, amb_type, chrg = 0.000001, layer = 'H'):
    """
    Based on atom object instance at_ins 
    generate and return a hydrogen link_atom_object
    ----
    at_ins : instance of atom object
    amb_type : string, amber atom type for H-link atom
    chrg : float, atomic charge to be ascribed to H-link atom
    """
    new_link_atom = link_atom( at_ins.get_coords(), at_ins.get_index(), 'H' )
    new_link_atom.set_connect_list( at_ins.get_connect_list() )
    new_link_atom.set_oniom_layer( layer )
    new_link_atom.set_type( amb_type )
    new_link_atom.set_at_charge( chrg )
    new_link_atom.set_mm_element( at_ins.get_element() )
    return new_link_atom 


def generate_label():
    label=''
    i = 0
    while True:
        j = int(i/26)
        k = i%26
        label = letters[k]
        if i > 0:
            for m in range( int(math.log(i,26)) ):
                k = j%26
                j = int(j/26)
                label = letters[k] + label
        i += 1
        yield label
        
        
def residue_within_r_from_atom(residue, atom, r):
    """checks if any atom of residue lies within r (d(at@res --- atom) <= r)
    of atom
    if yes, returns True, if no, returns False"""
    ref_coords = np.array(atom.get_coords(),dtype=float)
    for atom_from_resid in residue.get_atoms():
        coords = np.array(atom_from_resid.get_coords(),dtype=float)
        dist = scipy.spatial.distance.cdist(ref_coords,coords)
        if dist <= r:
            return True
    return False


def residues_within_r_from_atom(residue_list, atom, r):
    """checks if any atom of residue from a residue_list 
    lies within r (d(at@res --- atom) <= r) of atom
    if yes, returns a list with True value at corresponding index
    """
    if not atom:
        result = [True for i in range(len(residue_list))]
        return result
    result = []
    temp = []
    temp.append(atom.get_coords())
    ref_coords = np.array(temp,dtype=float)
    for residue in residue_list:
        coords = []
        for atom_from_resid in residue.get_atoms():
            coords.append(atom_from_resid.get_coords())
        coords_arr = np.array(coords,dtype=float)
        dist = scipy.spatial.distance.cdist(ref_coords,coords_arr)
        if np.amin(dist) <= r:
            result.append(True)
        else:
            result.append(False)
    return result


def residues_within_r_from_atom_list(residue_list, atom_list, r):
    """ 
    checks if any atom of residue from a residue_list 
    lies within r (d(at@res --- atom) <= r) of any atom from atom_list;
    Input:
        residue_list - a list of residue objects
        atom_list - a list of atom objects
        r - radius, float 
    Returns: 
        a boolean list of length len(residue_list)
    """
    res_list_length = len(residue_list)
    if len(atom_list) == 0:
        result = [True for i in range(res_list_length)]
        return result
    result = [False for i in range(res_list_length)]
    for atom_ref in atom_list:
        index_table = []
        temp_residue_list = []
        for i in range(res_list_length):
            if result[i] == False:
                temp_residue_list.append(residue_list[i])
                index_table.append(i)
        temp_result = residues_within_r_from_atom(temp_residue_list, atom_ref, r)
        for i in range(len(temp_residue_list)):
            result[index_table[i]] = (result[index_table[i]] or temp_result[i])
    return result

    
def nlayers_ONIOM(n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer):
    """
    based on number of atoms in H, M and L layers calculates how many layers 
    are in the ONIOM system

    Parameters
    ----------
    n_atom_in_H_layer : INT
        # of atoms in the H-layer.
    n_atom_in_M_layer : INT
        # of atoms in the M-layer.
    n_atom_in_L_layer : INT
        # of atoms in the L-layer.

    Returns
    -------
    nlayers : INT
        # of layers.

    """
    if (n_atom_in_H_layer > 0) and (n_atom_in_M_layer > 0) and (n_atom_in_L_layer > 0):
        nlayers = 3
    elif (n_atom_in_H_layer == 0) and (n_atom_in_M_layer == 0):
        nlayers = 1
    elif (n_atom_in_M_layer == 0) and (n_atom_in_L_layer == 0):
        nlayers = 1
    elif (n_atom_in_H_layer == 0) and (n_atom_in_L_layer == 0):
        nlayers = 1
    elif (n_atom_in_H_layer == 0) and (n_atom_in_L_layer == 0) and (n_atom_in_M_layer == 0):
        nlayers = 0
    else:
        nlayers = 2
    return nlayers
    

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


def charge_change(atom_list, link_atom_list, connect_dic, q_model) -> list:
    """
    Change atom charge and create a list of off-atom point charges at the QM/MM border
    Decisive which atoms get their charge changed is the link_atom_list
    :param atom_list: a list of atom objects - their charges can be changed
    :param link_atom_list: a list of link atom objects - their positions are decisive
    :param connect_dic: a dictionary with connectivity info (key: 0-based index, value: a list)
    :param q_model: string, type of charge model: "z1", "z2", "z3", "rc", "rcd" or "cs"
    :return: a list of off-atoms point charges - point_charge objects (may be empty)
    """
    off_atm_p_charges = []

    for lk_atom in link_atom_list:
        lk_atom_ix = lk_atom.get_index()
        m1_ix = lk_atom_ix
        m2_ixs = connect_dic[m1_ix]
        m2_ixs.remove(lk_atom.get_bonded_to())
        m3_ixs = []
        for m2ix in m2_ixs:
            m3_ixs += connect_dic[m2ix]
            m3_ixs.remove(m1_ix)
        
        m1_orig_q = atom_list[m1_ix].get_at_charge()
        atom_list[m1_ix].set_at_charge(0.0)
        
        if q_model == "z1":
            pass
        if (q_model == "z2") or (q_model == "z3"):
            for m2ix in m2_ixs:
                atom_list[m2ix].set_at_charge(0.0)
        if q_model == "z3":
            for m3ix in m3_ixs:
                atom_list[m3ix].set_at_charge(0.0)
        elif q_model == "rc":
            q = m1_orig_q/(len(m2_ixs))
            m1_coords = atom_list[m1_ix].get_coords()
            for m2ix in m2_ixs:
                m2_coords = atom_list[m2ix].get_coords()
                pq_coords = half_distance(m1_coords, m2_coords)
                off_atm_pq = point_charge(q, pq_coords)
                off_atm_p_charges.append(off_atm_pq)
        elif q_model == "rcd":
            q = m1_orig_q/(len(m2_ixs))
            m1_coords = atom_list[m1_ix].get_coords()
            for m2ix in m2_ixs:
                m2_coords = atom_list[m2ix].get_coords()
                pq_coords = half_distance(m1_coords, m2_coords)
                off_atm_pq = point_charge(2*q, pq_coords)
                off_atm_p_charges.append(off_atm_pq)
                m2_orig_q = atom_list[m2ix].get_at_charge()
                m2_new_q = m2_orig_q - q
                atom_list[m2ix].set_at_charge(m2_new_q)
        elif q_model == "cs":
            q = m1_orig_q/(len(m2_ixs))
            m1_coords = atom_list[m1_ix].get_coords()
            for m2ix in m2_ixs:
                m2_coords = atom_list[m2ix].get_coords()
                p1crd, p2crd = cs_q_coords(m1_coords, m2_coords)
                off_atm_pq_1 = point_charge(5*q, p1crd)
                off_atm_p_charges.append(off_atm_pq_1)
                off_atm_pq_2 = point_charge(-5*q, p2crd)
                off_atm_p_charges.append(off_atm_pq_2)
                m2_orig_q = atom_list[m2ix].get_at_charge()
                m2_new_q = m2_orig_q + q
                atom_list[m2ix].set_at_charge(m2_new_q)                

    return off_atm_p_charges


def half_distance(coords_1, coords_2) -> list:
    """
    calculate coords in half distance between two atoms
    :param coords_1: a list with 3 coordinates (p1)
    :param coords_2: a list with 3 coordinates (p2)
    :return: a list with coordinates of the middle point between p1 and p2
    """
    cor = [(coords_1[0] + coords_2[0]) / 2, (coords_1[1] + coords_2[1]) / 2, (coords_1[2] + coords_2[2]) / 2]
    return cor


def cs_q_coords(coords_1, coords_2) -> tuple:
    """
    calculate coords of additinal point charges in the "cs" scheme, around the M2 atom
    :param coords_1: a list - coordinates of atom M1
    :param coords_2: a list - coordinates of atom M2
    :return: a tuple of two lists, each with coordinates of a point charge
    """

    a = coords_1[0]-coords_2[0]
    b = coords_1[1]-coords_2[1]
    c = coords_1[2]-coords_2[2]

    
    p1_x = coords_2[0] + 0.1 * a
    p1_y = coords_2[1] + 0.1 * b
    p1_z = coords_2[2] + 0.1 * c
    p1_coords = [p1_x, p1_y, p1_z]
    
    p2_x = coords_2[0] - 0.1 * a
    p2_y = coords_2[1] - 0.1 * b
    p2_z = coords_2[2] - 0.1 * c
    p2_coords = [p2_x, p2_y, p2_z]   
    
    return p1_coords, p2_coords


def charge_summary(atom_list: list, link_atom_list: list, point_charge_list: list) -> tuple:
    """
    calculate sum of atoms' and point_charges' charges
    :param atom_list: a list of atom objects making the whole ONIOM system
    :param link_atom_list: a list of link atom objects
    :param point_charge_list: a list of point_charge objects
    :return: tuple (All_Q, H_Q, M_Q, L_Q, links_Q, p_charges_total)
    """
    All_Q = 0.0
    H_Q = 0.0
    M_Q = 0.0
    L_Q = 0.0
    links_Q = 0.0
    p_charges_total = 0.0
    
    for atom in atom_list:
        q = atom.get_at_charge()
        layer = atom.get_oniom_layer()
        All_Q += q
        if layer == "H":
            H_Q += q
        elif layer == "M":
            M_Q += q
        elif layer == "L":
            L_Q += q
        
    for lk_atom in link_atom_list:
        q = lk_atom.get_at_charge()
        All_Q += q
        links_Q += q
        
    for pq in point_charge_list:
        q = pq.get_charge()
        All_Q += q
        p_charges_total += q
        
    return All_Q, H_Q, M_Q, L_Q, links_Q, p_charges_total


def report_charges(q_summary):
    """
    prints results reported by charge_summary

    Parameters
    ----------
    q_summary : tuple
        (All_Q, H_Q, M_Q, L_Q, links_Q, p_charges_total)

    Returns
    -------
    None.

    """
    print("Total charge                       = " + str(round(q_summary[0], 5)))
    print("Total charge of H-layer            = " + str(round(q_summary[1], 5)))   
    print("Total charge of M-layer            = " + str(round(q_summary[2], 5)))
    print("Total charge of L-layer            = " + str(round(q_summary[3], 5)))
    print("Total charge of link atoms         = " + str(round(q_summary[4], 5)))
    print("Total charge of add. point charges = " + str(round(q_summary[5], 5)))


def count_atoms_in_layers(atom_obj_list, link_atom_obj_list):
    """
    calculates number of atoms in layers and number of link atoms

    Parameters
    ----------
    atom_obj_list : LIST 
        a list of atom objects.
    link_atom_obj_list : LIST
        a list og link atom objects
    Returns
    -------
    n_at_in_oniom : INT
        Total number of atoms in the list
    n_atom_in_H_layer : INT
        # of atoms in the H-layer
    n_atom_in_M_layer : INT
        # of atoms in the M-layer
    n_atom_in_L_layer : INT
        # of atoms in the L-layer
    n_link_atoms_for_H : INT
        # of link atoms capping the H-layer
    n_link_atoms_for_M : INT
        # of link atoms capping the M-layer    
    """
    n_at_in_oniom = len(atom_obj_list)
    n_atom_in_H_layer = 0
    n_atom_in_M_layer = 0
    n_atom_in_L_layer = 0
    n_link_atoms_for_H = 0
    n_link_atoms_for_M = 0
    n_LAH = 0
    for atom in atom_obj_list:
        if atom.get_oniom_layer() == "H":
            n_atom_in_H_layer += 1
        elif atom.get_oniom_layer() == "M":
            n_atom_in_M_layer += 1   
        elif atom.get_oniom_layer() == "L":
            n_atom_in_L_layer += 1  
        if atom.get_LAH(): # count link atom hosts
            n_LAH += 1
            
    for lk_atom in link_atom_obj_list:
        if lk_atom.get_oniom_layer() == 'H':
            n_link_atoms_for_H += 1
        elif lk_atom.get_oniom_layer() == 'M':
            n_link_atoms_for_M += 1
    
#    assert n_LAH == (n_link_atoms_for_H + n_link_atoms_for_M), "number of LAH != lk_for_H + lk_for_M" # ! multiple link atoms per single MM atom 
    return n_at_in_oniom, n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer,\
        n_link_atoms_for_H, n_link_atoms_for_M


def extract_at_atm_p_charges(atoms_list, layer="L"):
    """
    extract at atom point charges from the given layer 

    Parameters
    ----------
    atoms_list : list
        list of atom objects for the whole ONIOM system.
    layer : string, optional
        from which layer atomic charges shall be extracted. 
        Can be either "L", "M" or "H". The default is "L".

    Returns
    -------
    at_at_p_charges : list
        a list of point_charge objects.

    """
    at_at_p_charges = []
    for atom in atoms_list:
        atm_layer = atom.get_oniom_layer()
        if atm_layer == layer:
            atm_coords = atom.get_coords()
            atm_q = atom.get_at_charge()
            p_charge = point_charge(atm_q, atm_coords)
            at_at_p_charges.append(p_charge)
    return at_at_p_charges


def extract_qm_system(atoms_list, link_atom_list, layer="H"):
    """
    extracts a subsystem of ONIOM calculations: H + link atoms or M + link atoms

    Parameters
    ----------
    atoms_list : list
        a list of atom objects for the whole ONIOM system.
    link_atom_list : list
        a list of link atom objects for the whole ONIOM system.
    layer : string, optional
        which subsystem shall be extracted, "H" or "M". The default is "H".

    Returns
    -------
    qm_system_atoms : list
        a list of atom and link atom objects consisting the extracted subsystem.

    """
    qm_system_atoms = []
    for atom in atoms_list:
        atm_layer = atom.get_oniom_layer()
        if atm_layer == layer:
            qm_system_atoms.append(atom)
        elif atom.get_LAH():
            for link in link_atom_list:
                if link.get_index() == atom.get_index():
                    bd_to_ix = link.get_bonded_to()
                    if atoms_list[bd_to_ix].get_oniom_layer() == layer:
                        qm_system_atoms.append(link)
                        break
    return qm_system_atoms


def extract_chemical_composition(atoms_list):
    """
    extracts atom composition of H, M and L layers of an ONIOM system

    Parameters
    ----------
    atoms_list : list
        a list of atom objects for the whole ONIOM system.

    Returns
    -------
    H_layer_composition : list 
        a list of tuples (element, #atoms), e.g. (C, 18).
    M_layer_composition : list
        as above, for the M-layer
    L_layer_composition : list
        as above, for the L-layer

    """
    H_layer_composition = [] # a list of tuples (element, #atoms), e.g. (C, 18)
    M_layer_composition = []
    L_layer_composition = []
    H_layer_dic = {}
    M_layer_dic = {}
    L_layer_dic = {}
    for atom in atoms_list:
        atm_layer = atom.get_oniom_layer()
        atm_element = atom.get_element()
        if atm_layer == "H":
            if atm_element in H_layer_dic:
                H_layer_dic[atm_element] += 1
            else:
                H_layer_dic[atm_element] = 1
        elif atm_layer == "M":
            if atm_element in M_layer_dic:
                M_layer_dic[atm_element] += 1
            else:
                M_layer_dic[atm_element] = 1
        elif atm_layer == "L":
            if atm_element in L_layer_dic:
                L_layer_dic[atm_element] += 1
            else:
                L_layer_dic[atm_element] = 1
    
    for key in H_layer_dic.keys():
        H_layer_composition.append(tuple([key, H_layer_dic[key]]))
    for key in M_layer_dic.keys():
        M_layer_composition.append(tuple([key, M_layer_dic[key]]))
    for key in L_layer_dic.keys():
        L_layer_composition.append(tuple([key, L_layer_dic[key]]))
    
    return H_layer_composition, M_layer_composition, L_layer_composition
    

def mod_layer(residues, atoms, res_ix, schain_ix, ix, layer):
    """
    Modifies a layer attribute of atoms to value given by layer
    All atoms in residues whose (0-based) indexes are provided (res_ix) are affected
    All atoms in side chains of residues (schain_ix) are affected
    All atoms whose indexes (ix) are provided are affected

    Parameters
    ----------
    residues : LIST
        a list of residue objects (whole system).
    atoms : LIST
        a list of atom objects (whole system).
    res_ix : LIST
        a list of (0-based) residue indexes.
    schain_ix : LIST
        a list of (0-based) side chain (residue) indexes.
    ix : LIST
        a list of (0-based) atom indexes.
    layer : STRING
        symbol of an ONIOM layer, can be: 'H', 'M', or 'L'.

    Returns
    -------
    None.

    """
    for i in res_ix:
        res = residues[i]
        for at in res.get_atoms():
            at.set_oniom_layer(layer)

    for i in schain_ix:
        res = residues[i]
        for at in res.get_side_chain_atoms():
            at.set_oniom_layer(layer)   
            
    for i in ix:
        atoms[i].set_oniom_layer(layer)


def lk_atoms_mod(link_at_list, atom_list, layer):
    """
    Set "bonded_to" atribute of link atoms and
    calculate coordinates of H-link atom and sets "H_coords"
    attribute of link atoms

    Parameters
    ----------
    link_at_list : LIST
        a list of link atoms in the system.
    atom_list : LIST
        a list of all atoms in the system.
    layer : STRING
        symbol of a layer which is capped by these link atoms,
        can be 'H' or 'M'.

    Returns
    -------
    None.

    """
    for at in link_at_list:
        con_list = at.get_connect_list()
        for con_ix in con_list: # find QM atom bonded to a given link atom
            con_at = atom_list[con_ix]
            if con_at.get_oniom_layer() == layer:
                at.set_bonded_to(con_ix)
                adjust_HLA_coords(at, con_at)
                break


def print_help():  
    help_text = """A python3 script to read Gaussian ONIOM(QM:MM) input file and modify its content

Command line arguments:
    1. oniom input file name to read 
    2. name of the file to be produced (of type: oniom input, qm input, xyz)
    3. string switch, one from the list given below
    4. only for: "rag", "rqg", "rqq" and "omod" - name of an additional input file 
    (of type: xyz (rag and rqg), qout (rqq) or input (omod))

Meaning of the switches:
    eag  - extract xyz coordinates of all atoms and write into the xyz file
    eqg  - extract xyz coordinates of H-layer atoms and write into the xyz file
    ehmg - extract xyz coordinates of H- and M-layers atoms and write into the xyz file
    rag  - replace xyz coordinates of all atoms to those read from the xyz file 
    rqg  - replace xyz coordinates of H-layer atoms to those read from the xyz file
    rqq  - replace H-layer atom charges to those read from the qout (RESP) file
    z1   - prepare input for electronic embedding with the z1 charge model 
    z2   - prepare input for electronic embedding with the z2 charge model
    z3   - prepare input for electronic embedding with the z3 charge model
    rc   - prepare input for electronic embedding with the rc charge model
    rcd  - prepare input for electronic embedding with the rcd charge model
    cs   - prepare input for electronic embedding with the cs charge model
    wqm  - write QM-only Gaussian input
    wqm_z1/z2/z3/rc/rcd/cs - write QM-only Gaussian input for ESP(RESP) calculations
    omod - modify oniom partitioning (2 or 3-layered) and/or frozen/optimized zone"""
    
    print(help_text)
    

def sum_p_charges(pq_list):
    """
    For a list of point_charge objects calculate a total charge

    Parameters
    ----------
    pq_list : LIST
        A list of point_charge objects.

    Returns
    -------
    Q_tot : FLOAT
        Total charge (of point charges in the input list)

    """
    Q_tot = 0.0
    for pq in pq_list:
        Q_tot += pq.get_charge()
    return Q_tot
