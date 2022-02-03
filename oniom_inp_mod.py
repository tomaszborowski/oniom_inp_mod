# -*- coding: utf-8 -*-
"""
oniom_inp_mod

A python3 script to read Gaussian ONIOM(QM:MM) input file and modify its content

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
    omod - modify oniom partitioning (2 or 3-layered)

authors: Jakub Baran, Paulina Miśkowiec, Tomasz Borowski
"""
import sys, os, re
from copy import deepcopy


from oniom_inp_mod_aux import find_in_file, read_from_to_empty_line, read_charge_spin
from oniom_inp_mod_aux import read_atom_inf, read_connect_list, write_xyz_file
from oniom_inp_mod_aux import read_xyz_file, read_qout_file, count_atoms_in_layers
from oniom_inp_mod_aux import write_oniom_inp_file, write_qm_input, charge_change
from oniom_inp_mod_aux import extract_at_atm_p_charges, extract_qm_system, extract_chemical_composition
from oniom_inp_mod_aux import vdw_radii, charge_summary, report_charges
#from oniom_inp_mod_aux import 


# Important variables (switches):
nlayers = 2 # number of layers in the ONIOM (2 or 3)



### ---------------------------------------------------------------------- ###
### test cases                                                             ###
# oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_17_07_b2.com"
# output_fname = 'input_examples/test_out'
# add_inp_fname = 'input_examples/h6h-oxo+succinate+water_hyo_17_07_b_moved.xyz'
# switch = 'rag'

# oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_17_07_b2.com"
# output_fname = 'input_examples/test_out'
# add_inp_fname = 'input_examples/H_layer_mod.xyz'
# switch = 'rqg'

# oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_17_07_b2.com"
# output_fname = 'input_examples/test_out'
# add_inp_fname = 'input_examples/fake.qout'
# switch = 'rqq'

oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_17_07_b2.com"
output_fname = 'input_examples/test_out'
switch = 'wqm_z1'

# oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_17_07_b2.com"
# output_fname = 'input_examples/test_out'
# switch = 'z1'

# resp_qout_file_to_read = 'input_examples/h6h-oxo+succinate+water_hyo_rep3_copy3_clust3-resp2.qout'
# whole_system_xyz_file_to_read = 'input_examples/h6h-oxo+succinate+water_hyo_17_07_b_moved.xyz'
# qm_system_xyz_file_to_read = ''




### ---------------------------------------------------------------------- ###
### Seting the file names                                                  ###
# oniom_inp = sys.argv[1]
# output_fname = sys.argv[2]
# switch = sys.argv[3]
# if len(sys.argv) > 4:
#     add_inp_fname = sys.argv[4]

# LEGAL_SWITCHES = ["eag", "eqg", "ehmg", "rag", "rqg", "rqq", "z1", "z2", "z3",\
#                   "rc", "rcd", "cs", "omod"]

# if switch not in LEGAL_SWITCHES:
#     print("Provided switch: ", switch, " was not recognized\n")
#     exit(1)

# if not os.path.isfile(oniom_inp):
#     print("ONIOM input file not found \n")
#     exit(1)

# if len(sys.argv) > 4:
#     if not os.path.isfile(add_inp_fname):
#         print("additional input file not found \n")
#         exit(1)


### ---------------------------------------------------------------------- ###
### Reading from the oniom_inp_to_read file                                ###

oniom_inp_f = open(oniom_inp, 'r')
inp_offsets = find_in_file(oniom_inp_f)


inp_header = read_from_to_empty_line(oniom_inp_f, inp_offsets["header"])

inp_comment = read_from_to_empty_line(oniom_inp_f, inp_offsets["comment"])

inp_charge_and_spin = read_charge_spin(oniom_inp_f, inp_offsets["chargeAndSpin"])

inp_atoms_list, inp_link_atoms_list, inp_H_and_LAH_index_list = \
    read_atom_inf(oniom_inp_f, inp_offsets["atomInfo"])


if inp_offsets["connectList"] > 0:
    inp_connect = read_connect_list(oniom_inp_f, inp_offsets["connectList"])
else:
    print("Connectivity section not found\n")

inp_redundant = None
if inp_offsets["redundant"] > 0:
    inp_redundant = read_from_to_empty_line(oniom_inp_f, inp_offsets["redundant"])
    print("Redundant coordinates section found\n")

inp_params = None
if inp_offsets["parm"] > 0:
    inp_params = read_from_to_empty_line(oniom_inp_f, inp_offsets["parm"])
else:
    print("FF parameters section not found\n")

oniom_inp_f.close()


### ---------------------------------------------------------------------- ###
### CASE: extract xyz coordinates and write into xyz file                  ###
if switch in ["eag", "eqg", "ehmg"]:
    output_f = open(output_fname, 'a')
    if switch == "eag":
        layer = "HML"
    elif switch == "eqg":
        layer = "H"
    elif switch == "ehmg":
        layer = "HM"

    write_xyz_file(output_f, inp_atoms_list, inp_link_atoms_list, layer)
    output_f.close()
    

### ---------------------------------------------------------------------- ###
### CASE: replace geometry for that read from additional xyz file          ###
if switch in ["rag", "rqg"]:
    xyz_f = open(add_inp_fname, 'r')
    xyz_atom_list = read_xyz_file(xyz_f)
    xyz_f.close()
    
    mod_atoms_list = deepcopy(inp_atoms_list)
    
    n_at_in_xyz = len(xyz_atom_list)
    n_at_in_oniom, n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer,\
        n_link_atoms_for_H = count_atoms_in_layers(mod_atoms_list)


    if (n_atom_in_M_layer > 0) and (n_atom_in_L_layer > 0):
        nlayers = 3
    
    if switch == "rag":
        if n_at_in_xyz != n_at_in_oniom:
            print("number of atoms read from oniom input file: ", n_at_in_oniom)
            print("\ndoes not match that read from the xyz input file: ", n_at_in_xyz)
            exit(1)
        for atom, xyz_line in zip(mod_atoms_list, xyz_atom_list):
            assert atom.get_element() == xyz_line[0], "element of atom " + str(atom.get_index()) + " does not match"
            atom.set_coords(xyz_line[1:4])
            
    elif switch == "rqg":
        if n_at_in_xyz != (n_atom_in_H_layer + n_link_atoms_for_H):
            print("number of H-layer + H-link atoms read from oniom input file: ", n_atom_in_H_layer + n_link_atoms_for_H)
            print("\ndoes not match that read from the xyz input file: ", n_at_in_xyz)
            exit(1)
        count = 0    
        for atom in mod_atoms_list:
            if (atom.get_oniom_layer() == "H"): 
                assert atom.get_element() == xyz_atom_list[count][0], "element of atom " + str(atom.get_index()) + " does not match"
                atom.set_coords(xyz_atom_list[count][1:4])
                count += 1
            elif (atom.get_oniom_layer() == "L") and atom.get_LAH(): # wymaga uogólnienia na przypadek gdy H/M/L
                assert "H" == xyz_atom_list[count][0], "atom " + str(atom.get_index()) + "should be H " + str(count) + " in the xyz file "  
                count += 1
                
    out_file = open(output_fname, 'a')
    write_oniom_inp_file(out_file, inp_header, inp_comment, inp_charge_and_spin, nlayers,\
                         mod_atoms_list, inp_link_atoms_list, inp_connect, inp_redundant, inp_params)    
    out_file.close()


### ---------------------------------------------------------------------- ###
### CASE: replace atomic charges of H-layer atoms to those read from qout  ###
if switch == "rqq":
    qout_f = open(add_inp_fname, 'r')
    qm_system_new_q = read_qout_file(qout_f)
    qout_f.close()
    
    mod_atoms_list = deepcopy(inp_atoms_list)

    n_q_in_qout = len(qm_system_new_q)
    n_at_in_oniom, n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer,\
        n_link_atoms_for_H = count_atoms_in_layers(mod_atoms_list)    

    if n_q_in_qout != (n_atom_in_H_layer + n_link_atoms_for_H):
        print("number of H-layer + H-link atoms read from oniom input file: ", n_atom_in_H_layer + n_link_atoms_for_H)
        print("\ndoes not match number of charges read from the qout input file: ", n_q_in_qout)
        exit(1)

    count = 0    
    for atom in mod_atoms_list:
        if (atom.get_oniom_layer() == "H"): 
            atom.set_at_charge(qm_system_new_q[count])
            count += 1
        elif (atom.get_oniom_layer() == "L") and atom.get_LAH(): # wymaga uogólnienia na przypadek gdy H/M/L
            count += 1    # skip LAH / link atom

    out_file = open(output_fname, 'a')
    write_oniom_inp_file(out_file, inp_header, inp_comment, inp_charge_and_spin, nlayers,\
                         mod_atoms_list, inp_link_atoms_list, inp_connect, inp_redundant, inp_params)    
    out_file.close()    


### ---------------------------------------------------------------------- ###
### CASE: write QM-only Gaussian input                                     ###
if switch in ["wqm", "wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:
    read_radii = False # if additional radii info will be placed in the Gaussian input file
    off_atm_p_q = []
    at_atm_p_q = [] 
    qm_system_atoms = []
    sum_charges_initial = charge_summary(inp_atoms_list, inp_link_atoms_list, [])
    print("\nTotal charges read from the input file:")
    report_charges(sum_charges_initial)
    if switch in ["wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:
        mod_atoms_list = deepcopy(inp_atoms_list)
        off_atm_p_q = charge_change(mod_atoms_list, inp_link_atoms_list, inp_connect, switch)
        at_atm_p_q = extract_at_atm_p_charges(mod_atoms_list, layer="L")
        qm_system_atoms = extract_qm_system(mod_atoms_list, inp_link_atoms_list, layer="H")
    elif switch == "wqm":
        qm_system_atoms = extract_qm_system(inp_atoms_list, inp_link_atoms_list, layer="H")

    all_point_charges = at_atm_p_q + off_atm_p_q
    
    if switch in ["wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:  
        sum_charges_final = charge_summary(mod_atoms_list, inp_link_atoms_list, off_atm_p_q)
        print("\nTotal charges after charge modification:")
        report_charges(sum_charges_final)

    resp_header_read_radii =  "%chk=name.chk\n" +\
                    "%Nproc=24\n" +\
                    "%Mem=24GB\n" +\
                    "# UB3LYP/def2SVP 5d scf=(xqc,maxcycle=350) charge\n" +\
                    "nosymm Pop=(MK,ReadRadii) iop(6/33=2) iop(6/42=6) iop(6/50=1)\n"

    resp_header =  "%chk=name.chk\n" +\
                    "%Nproc=24\n" +\
                    "%Mem=24GB\n" +\
                    "# UB3LYP/def2SVP 5d scf=(xqc,maxcycle=350) charge\n" +\
                    "nosymm Pop=(MK) iop(6/33=2) iop(6/42=6) iop(6/50=1)\n"

    qm_header =     "%chk=name.chk\n" +\
                    "%Nproc=24\n" +\
                    "%Mem=24GB\n" +\
                    "# UB3LYP/def2SVP 5d scf=(xqc,maxcycle=350) nosymm\n"
    
    if switch in ["wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:
        H_layer_composition, M_layer_composition, L_layer_composition = extract_chemical_composition(mod_atoms_list)
    elif switch == "wqm":
        H_layer_composition, M_layer_composition, L_layer_composition = extract_chemical_composition(inp_atoms_list)

    read_radii_lines = ""
    
    for item in H_layer_composition:
        ele = item[0]
        for key in vdw_radii.keys():
            if key == ele:
                read_radii = True
                line = str(key) + "   " + str(vdw_radii[key]) + "\n"
                read_radii_lines += line               
    
    out_file = open(output_fname, 'a')
    
    if switch in ["wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:
        comment = "QM/MM charge model: " + switch + "\n"
        if read_radii:
            write_qm_input(out_file, resp_header_read_radii, comment, inp_charge_and_spin, qm_system_atoms, all_point_charges)
            out_file.write(read_radii_lines)
        else:
            write_qm_input(out_file, resp_header, comment, inp_charge_and_spin, qm_system_atoms, all_point_charges)
        gesp_f_name = output_fname + ".gesp\n"
        out_file.write("\n")
        out_file.write(gesp_f_name)
        out_file.write("\n")
        out_file.write(gesp_f_name)
        out_file.write("\n")
    elif switch == "wqm":
        comment = "QM system from: " + oniom_inp + "\n"
        write_qm_input(out_file, qm_header, comment, inp_charge_and_spin, qm_system_atoms, [])

    out_file.close()




