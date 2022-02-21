#!/usr/bin/env python3
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
    omod - modify oniom partitioning (2 or 3-layered) and/or frozen/optimized zone

authors: Jakub Baran, Paulina Miśkowiec, Tomasz Borowski
last update: 17 Feb 2022
"""
import sys, os, re
from copy import deepcopy
import numpy as np

from oniom_inp_mod_aux import find_in_file, read_from_to_empty_line, read_charge_spin,\
read_atom_inf, read_connect_list, read_p_charges, write_xyz_file, atom_to_link_atom,\
read_xyz_file, read_qout_file, count_atoms_in_layers, adjust_HLA_coords, print_help,\
write_mm_inp_file, write_oniom_inp_file, write_qm_input, charge_change, sum_p_charges,\
extract_at_atm_p_charges, extract_qm_system, extract_chemical_composition,write_mm_input,\
vdw_radii, charge_summary, report_charges, nlayers_ONIOM,\
read_single_string, read_single_number, read_pdb_file,\
read_rsi_index, input_read_link_atoms, input_read_freeze,\
residue, main_side_chain, mod_layer, write_pdb_file,\
lk_atoms_mod, generate_label, peptide, N_CO_in_residue, is_peptide_bond2


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

# oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_17_07_b2.com"
# output_fname = 'input_examples/test_out'
# switch = 'wqm_z1'

# oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_17_07_b2.com"
# output_fname = 'input_examples/test_out'
# switch = 'cs'

# resp_qout_file_to_read = 'input_examples/h6h-oxo+succinate+water_hyo_rep3_copy3_clust3-resp2.qout'
# whole_system_xyz_file_to_read = 'input_examples/h6h-oxo+succinate+water_hyo_17_07_b_moved.xyz'
# qm_system_xyz_file_to_read = ''

# oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo.com"
# output_fname = 'input_examples/test_out'
# add_inp_fname = 'input_examples/omod.inp'
# switch = 'omod'

#oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo.com"
#oniom_inp = "input_examples/h6h-oxo+succinate+water_hyo_fake_pq.com"
#output_fname = 'input_examples/test_out'
#add_inp_fname = 'input_examples/omod_no_freeze.inp'
#add_inp_fname = 'input_examples/omod_nf_no_lk.inp'
#switch = 'omod'

#oniom_inp = "-h"

### ---------------------------------------------------------------------- ###
### Seting the file names                                                  ###
sys_argv_len = len(sys.argv)
if sys_argv_len > 1:
    oniom_inp = sys.argv[1]
else:
    oniom_inp = None
if sys_argv_len > 2:
    output_fname = sys.argv[2]
else:
    output_fname = None
if sys_argv_len > 3:
    switch = sys.argv[3]
else:
    switch = None
if sys_argv_len > 4:
    add_inp_fname = sys.argv[4]
else:
    add_inp_fname = None

LEGAL_SWITCHES = ["eag", "eqg", "ehmg", "rag", "rqg", "rqq", "z1", "z2", "z3",\
		  "rc", "rcd", "cs", "wqm", "wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc",\
		  "wqm_rcd", "wqm_cs", "omod"]

### if -h - write help and exit                                            ###
if oniom_inp == "-h":
    print_help()
    sys.exit(1)
    
if switch not in LEGAL_SWITCHES:
    print("Provided switch: ", switch, " was not recognized\n")
    sys.exit(1)

if not os.path.isfile(oniom_inp):
    print("ONIOM input file not found \n")
    sys.exit(1)

if not output_fname:
    print("Output file name is required as the 2nd argument \n")
    sys.exit(1)    

if len(sys.argv) > 4:
    if not os.path.isfile(add_inp_fname):
       print("additional input file not found \n")
       sys.exit(1)

print("#----------------------------------------------------------------------#")
print("input file: ", oniom_inp)
print("output file: ", output_fname)
print("switch: ", switch)
if add_inp_fname:
    print("additional input file name: ", add_inp_fname)

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

#inp_p_charges = None
if inp_offsets["p_charges"] > 0:
    inp_p_charges = read_p_charges(oniom_inp_f, inp_offsets["p_charges"])
    print("Point charge section found")
    print("Number of point charges read: ", len(inp_p_charges))
    inp_pq_sum = sum_p_charges(inp_p_charges)
    print("Their total charge = ", str( round(inp_pq_sum, 8) ) )
else:
    inp_p_charges = []
    
oniom_inp_f.close()


### ---------------------------------------------------------------------- ###
### set appropriate (H or M) oniom_layer for link atoms                    ###
for lk_at in inp_link_atoms_list:
    bonded_to = lk_at.get_bonded_to()
    b2_layer = inp_atoms_list[bonded_to].get_oniom_layer()
    lk_at.set_oniom_layer(b2_layer)


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

    print("#----------------------------------------------------------------#")
    print("Extracting ", layer, " layer(s) (+ link atoms) into the xyz file\n")
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
        n_link_atoms_for_H, n_link_atoms_for_M = count_atoms_in_layers(mod_atoms_list, inp_link_atoms_list)

    nlayers = nlayers_ONIOM(n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer)

    print("#-------------------------------------------------------------#")
    if switch == "rag":
        print("Replacing geometry of the whole system to that read from the xyz file\n")
        if n_at_in_xyz != n_at_in_oniom:
            print("number of atoms read from oniom input file: ", n_at_in_oniom)
            print("\ndoes not match that read from the xyz input file: ", n_at_in_xyz)
            exit(1)
        for atm, xyz_line in zip(mod_atoms_list, xyz_atom_list):
            assert atm.get_element() == xyz_line[0], "element of atom " + str(atm.get_index()) + " does not match"
            atm.set_coords(xyz_line[1:4])
            
    elif switch == "rqg":
        print("Replacing geometry of the H-layer to that read from the xyz file\n")
        if n_at_in_xyz != (n_atom_in_H_layer + n_link_atoms_for_H):
            print("number of H-layer + H-link atoms read from oniom input file: ", n_atom_in_H_layer + n_link_atoms_for_H)
            print("\ndoes not match that read from the xyz input file: ", n_at_in_xyz)
            exit(1)
        count = 0    
        for atm in mod_atoms_list:
            if (atm.get_oniom_layer() == "H"): 
                assert atm.get_element() == xyz_atom_list[count][0], "element of atom " + str(atm.get_index()) + " does not match"
                atm.set_coords(xyz_atom_list[count][1:4])
                count += 1
            elif (atm.get_oniom_layer() == "L") and atm.get_LAH(): # wymaga uogólnienia na przypadek gdy H/M/L
                assert "H" == xyz_atom_list[count][0], "atom " + str(atm.get_index()) + "should be H " + str(count) + " in the xyz file "  
                count += 1
                
    out_file = open(output_fname, 'a')
    write_oniom_inp_file(out_file, inp_header, inp_comment, inp_charge_and_spin, nlayers,\
                         mod_atoms_list, inp_link_atoms_list, inp_connect, inp_redundant, inp_params)    
    out_file.close()


### ---------------------------------------------------------------------- ###
### CASE: replace atomic charges of H-layer atoms to those read from qout  ###
if switch == "rqq":
    print("#----------------------------------------------------------------------#")
    print("Changing atomic charges of H-layer atoms to values read from qout file\n")
    
    mod_atoms_list = deepcopy(inp_atoms_list)
    
    n_at_in_oniom, n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer,\
        n_link_atoms_for_H, n_link_atoms_for_M = count_atoms_in_layers(mod_atoms_list, inp_link_atoms_list)    
    nlayers = nlayers_ONIOM(n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer)
    
    qout_f = open(add_inp_fname, 'r')
    qm_system_new_q = read_qout_file(qout_f)
    qout_f.close()

    n_q_in_qout = len(qm_system_new_q)
    qout_Q_total = np.sum( np.array(qm_system_new_q) )
    print("From qout file read: ", n_q_in_qout, "charge values")
    print("Their sum = ", str( round(qout_Q_total, 6) ) )
    
    
    if n_q_in_qout != (n_atom_in_H_layer + n_link_atoms_for_H):
        print("\nNumber of H-layer + H-link atoms read from oniom input file: ", n_atom_in_H_layer + n_link_atoms_for_H)
        print("does not match number of charges read from the qout input file: ", n_q_in_qout)
        exit(1)

    sum_charges_initial = charge_summary(inp_atoms_list, inp_link_atoms_list, inp_p_charges)
    print("\nTotal charges read from the input ONIOM file:")
    report_charges(sum_charges_initial)

    count = 0    
    lk_tot_charge = 0
    for atm in mod_atoms_list:
        if (atm.get_oniom_layer() == "H"): 
            atm.set_at_charge(qm_system_new_q[count])
            count += 1
        elif atm.get_LAH():
            at_ix = atm.get_index()
            for lk_at in inp_link_atoms_list:
                if (lk_at.get_index() == at_ix):
                    if (lk_at.get_oniom_layer() == "H"):
                        lk_tot_charge += qm_system_new_q[count]
                        count += 1    # skip LAH / link atom   

    sum_charges = charge_summary(mod_atoms_list, inp_link_atoms_list, inp_p_charges)
    print("\nTotal charges after ascribing H-layer charges read from the qout file:")
    report_charges(sum_charges)
    print("Total charge not ascribed (to LAHs) = ", str(round(lk_tot_charge, 6)))

    out_file = open(output_fname, 'a')
    write_oniom_inp_file(out_file, inp_header, inp_comment, inp_charge_and_spin, nlayers,\
                         mod_atoms_list, inp_link_atoms_list, inp_connect, inp_redundant, inp_params)    
    out_file.close()    


### ---------------------------------------------------------------------- ###
### CASE: write QM-only Gaussian input                                     ###
if switch in ["wqm", "wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:
    print("#----------------------------------------------------------------------------------------------#")
    if switch == "wqm":
        print("Writing QM Gaussian input file for the H-layer\n")
    else:
        print("Writing QM Gaussian for RESP input file with point charges within the model: ", switch[4:0], "\n")

        n_at_in_oniom, n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer,\
            n_link_atoms_for_H, n_link_atoms_for_M = count_atoms_in_layers(inp_atoms_list, inp_link_atoms_list)
        nlayers = nlayers_ONIOM(n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer)  

        mod_atoms_list = deepcopy(inp_atoms_list)
        mod_link_atoms_list = deepcopy(inp_link_atoms_list)
        if (n_atom_in_M_layer > 0) and (nlayers == 3):
            print("WARNING: 3-layer ONIOM system read, M-layer charges will be treated as L-layer charges\n")
            nlayers = 2

            for atm in mod_atoms_list:
                atm_ix = atm.get_index()
                if atm.get_oniom_layer() == 'M':
                    atm.set_oniom_layer('L')
                elif ( (atm.get_oniom_layer() == 'L') and atm.get_LAH() ):
                    for lk in mod_link_atoms_list.copy():
                        if lk.get_index() == atm_ix:
                            mod_link_atoms_list.remove(lk)

    read_radii = False # if additional radii info will be placed in the Gaussian input file
    off_atm_p_q = []
    at_atm_p_q = [] 
    qm_system_atoms = []
    sum_charges_initial = charge_summary(inp_atoms_list, inp_link_atoms_list, inp_p_charges)
    print("\nTotal charges read from the input file:")
    report_charges(sum_charges_initial)
    if switch in ["wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:
        q_model = switch[4:]
        off_atm_p_q = charge_change(mod_atoms_list, mod_link_atoms_list, inp_connect, q_model)
        at_atm_p_q = extract_at_atm_p_charges(mod_atoms_list, layer="L")
        qm_system_atoms = extract_qm_system(mod_atoms_list, mod_link_atoms_list, layer="H")
    elif switch == "wqm":
        qm_system_atoms = extract_qm_system(inp_atoms_list, inp_link_atoms_list, layer="H")

    all_point_charges = at_atm_p_q + off_atm_p_q
    
    if switch in ["wqm_z1", "wqm_z2", "wqm_z3", "wqm_rc", "wqm_rcd", "wqm_cs" ]:  
        sum_charges_final = charge_summary(mod_atoms_list, mod_link_atoms_list, off_atm_p_q)
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
        out_file.write(gesp_f_name)
        out_file.write("\n")
        out_file.write(gesp_f_name)
        out_file.write("\n")
    elif switch == "wqm":
        comment = "QM system from: " + oniom_inp + "\n"
        write_qm_input(out_file, qm_header, comment, inp_charge_and_spin, qm_system_atoms, [])

    out_file.close()


### ---------------------------------------------------------------------- ###
### CASE: write ONIOM=EE Gaussian input                                    ###
if switch in ["z1", "z2", "z3", "rc", "rcd", "cs" ]:
    print("#---------------------------------------------------------------------------------#")
    print("Writing 2-layer ONIOM=ElectronicEmbeding input file with charge QM-MM model: ", switch, "\n")

    find = re.search(r'[Oo][Nn][Ii][Oo][Mm]\((.+)\)', inp_header)
    if find:
        old_o_command = find.group(0)       
    else:
        old_o_command = None

    find2 = re.search(r'=[Ee][Mm][Bb][Ee][Dd][Cc][Hh][Aa][Rr][Gg][Ee]', inp_header)
    if find2:
        old_ee = find2.group(0)
    else:
        old_ee = None
        
    find3 = re.search(r'=[Ss][Cc][Aa][Ll][Ee][Cc][Hh][Aa][Rr][Gg][Ee]=\d{1,6}', inp_header)
    if find3:
        old_scalecharge = find3.group(0)
    else:
        old_scalecharge = None    

    if switch in ["rc", "rcd", "cs" ]:
        new_scalecharge = "=ScaleCharge=555555 charge "
    elif switch == "z1":
        new_scalecharge = "=ScaleCharge=555555 "
    elif switch == "z2":
        new_scalecharge = "=ScaleCharge=555550 "
    elif switch == "z3":
        new_scalecharge = "=ScaleCharge=555500 "

    if old_scalecharge:
        mod_header = inp_header.replace(old_scalecharge, new_scalecharge)
    elif old_ee:
        mod_header = inp_header.replace(old_ee, new_scalecharge)
    elif old_o_command:
        new_o_command = old_o_command + new_scalecharge
        mod_header = inp_header.replace(old_o_command, new_o_command)

    n_at_in_oniom, n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer,\
        n_link_atoms_for_H, n_link_atoms_for_M = count_atoms_in_layers(inp_atoms_list, inp_link_atoms_list)
    nlayers = nlayers_ONIOM(n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer)  

    mod_atoms_list = deepcopy(inp_atoms_list)
    mod_link_atoms_list = deepcopy(inp_link_atoms_list)
    if (n_atom_in_M_layer > 0) and (nlayers == 3):
        print("WARNING: 3-layer ONIOM system read, 2-layer (H/L) ONIOM system will be written\n")
        nlayers = 2
# zmodyfikować o_command - usunąć wpis o medium metodzie obliczeniowej
        for atm in mod_atoms_list:
            atm_ix = atm.get_index()
            if atm.get_oniom_layer() == 'M':
                atm.set_oniom_layer('L')
            elif ( (atm.get_oniom_layer() == 'L') and atm.get_LAH() ):
                for lk in mod_link_atoms_list.copy():
                    if lk.get_index() == atm_ix:
                        mod_link_atoms_list.remove(lk)
        
    comment = "QM/MM charge model: " + switch + "\n"
    off_atm_p_q = []
    sum_charges_initial = charge_summary(inp_atoms_list, inp_link_atoms_list, inp_p_charges)
    print("\nTotal charges read from the input file:")
    report_charges(sum_charges_initial)

    mod_Q_atoms_list = deepcopy(mod_atoms_list)
    off_atm_p_q = charge_change(mod_Q_atoms_list, mod_link_atoms_list, inp_connect, switch)
    if len(inp_p_charges) > 0:
        print("\nWARNING: point charges read from the input file will be retained and added to off-atom\
              point charges")
        off_atm_p_q += inp_p_charges
    
    sum_charges_final = charge_summary(mod_Q_atoms_list, mod_link_atoms_list, off_atm_p_q)
    print("\nTotal charges after charge modification or in ONIOM=ScaleCharge calculations (z1, z2 or z3):")
    report_charges(sum_charges_final)

    at_atm_p_q = extract_at_atm_p_charges(mod_Q_atoms_list, layer="L")
    all_point_charges = at_atm_p_q + off_atm_p_q

    out_file = open(output_fname, 'a')
    if switch in ["z1", "z2", "z3" ]:
        print("\nIn the written file atom partial charges are not modified, only the ScaleCharge option")
        write_oniom_inp_file(out_file, mod_header, comment, inp_charge_and_spin, nlayers,\
                         mod_atoms_list, mod_link_atoms_list, inp_connect,\
                         inp_redundant, inp_params, off_atm_p_q)

    elif switch in ["rc", "rcd", "cs" ]:
        # set connectivity to atoms based on connectivity list read from the ONIOM input
        for at in mod_Q_atoms_list:
            at_ix = at.get_index()
            connect = inp_connect[at_ix]
            at.set_connect_list(connect)
            
        qm_system_atoms = extract_qm_system(mod_Q_atoms_list, mod_link_atoms_list, layer="H")
        
        if inp_redundant:
            print("\nWARNING: redundant section read from the input file is dropped")
        out_redundant = None 
        
        mm_real_comment = 'MM calculations for the real system\n'
        qm_model_comment = 'QM calculations for the model system\n'
        mm_model_comment = 'MM calculations for the model system\n'
        
        find = re.search(r'[Oo][Nn][Ii][Oo][Mm]\(([0-9A-Za-z/:=]+)([ ]*\))', inp_header)
        if find:
            qm_method = find.group(1).split(':')[0]
            mm_method = find.group(1).split(':')[-1]
        else:
            qm_method = 'None'
            mm_method = 'None'
            print("\nWARNING: qm and mm methods not found, you must modify the output yourself")

        qm_header = inp_header.replace(find.group(0), qm_method)
        mm_header = inp_header.replace(find.group(0), mm_method)

        find = re.search(r'[Oo][Pp][Tt](([0-9A-Za-z,=\(\)]+)([ ]*))', inp_header)
        if find:
            qm_header = qm_header.replace(find.group(0), '')
            mm_header = mm_header.replace(find.group(0), '')

        find = re.search(r'[Ss][Cc][Ff][=]\(([0-9A-Za-z,=]+)([ ]*\))', mm_header)
        if find:
            mm_header = mm_header.replace(find.group(0), '')
            
        find = re.search(r'[Cc][Hh][Aa][Rr][Gg][Ee]', inp_header)
        if not find:
            qm_header = qm_header + ' charge'
            mm_model_header = mm_header + ' charge'
        
        qm_header = qm_header.replace('\n', ' ')
        mm_header = mm_header.replace('\n', ' ')
        mm_model_header = mm_model_header.replace('\n', ' ')
        
        qm_header += '\n'
        mm_header += '\n'
        mm_model_header += '\n'
        
        write_mm_inp_file(out_file, mm_header, mm_real_comment, inp_charge_and_spin, mod_atoms_list,\
                          inp_connect, out_redundant, inp_params, inp_p_charges)
        out_file.write("--Link1--\n")
        write_qm_input(out_file, qm_header, qm_model_comment, inp_charge_and_spin, qm_system_atoms, all_point_charges)
        out_file.write("--Link1--\n")
        write_mm_input(out_file, mm_model_header, mm_model_comment, inp_charge_and_spin, qm_system_atoms, all_point_charges, inp_params)

    out_file.close()


### -------------------------------------------------------------------------- ###
### CASE: read separate input file (to modify the ONIOM system partitioning)   ###
if switch == "omod":
    input_f = open(add_inp_fname, 'r')
    print("#---------------------------------------------------------------------------------#")
    print("Modifying the ONIOM system partitioning according to info read from file: ", add_inp_fname, "\n")
    print("Content of this file: \n")
    content_inp_f = input_f.read()
    print(content_inp_f)
    print("#---------------------------------------------------------------------------------#")
    mod_atoms_list = deepcopy(inp_atoms_list)

# inform about charges in the input ONIOM file
    sum_charges_initial = charge_summary(inp_atoms_list, inp_link_atoms_list, inp_p_charges)
    print("\nTotal charges read from the ONIOM input file:")
    report_charges(sum_charges_initial)

# set connectivity to atoms based on connectivity list read from the ONIOM input
    for at in mod_atoms_list:
        at_ix = at.get_index()
        connect = inp_connect[at_ix]
        at.set_connect_list(connect)
    
    pdb_file_name = read_single_string(input_f, "%pdb_f_name")
    qH = read_single_number(input_f, "%H_charge")
    mH = read_single_number(input_f, "%H_multip")
    qM = read_single_number(input_f, "%M_charge")
    mM = read_single_number(input_f, "%M_multip")
    qL = read_single_number(input_f, "%L_charge")
    mL = read_single_number(input_f, "%L_multip")

#   read pdb and create residue list
    pdb_f = open(pdb_file_name, 'r')
    res_list_from_pdb, at_list_from_pdb = read_pdb_file(pdb_f)
    pdb_f.close()

    if len(mod_atoms_list) != len(at_list_from_pdb):
        input_f.close()
        print("\n Number of atoms in the inp and pdb files do not match\n")
        print("# at in the ONIOM input: ", str(len(mod_atoms_list)))
        print("\n# at in the PDB file: ", str(len(at_list_from_pdb)))
        exit(1)

#   take info about residues from PDB and asscribe atoms read from ONIOM input
#   into residues (the order of atoms in the two input files must be the same !)
    residues = []
    i = 0
    for pdb_res in res_list_from_pdb:
        label = pdb_res.get_label()
        index = pdb_res.get_index()
        new_res = residue(label, index)
        n_atms = len(pdb_res.get_atoms())
        for j in range(n_atms):
            new_res.add_atom(mod_atoms_list[i+j])
        residues.append(new_res)
        i += n_atms

# process residues to set in_mainchain atribute of atoms (protein main chain)
# and populate main_chain_atoms atribute of residues    
    for res in residues:
        main_side_chain(res)
        res.set_new_index(res.get_index())

# erase all info about ONIOM layers read from the ONIOM input file (all atoms -> 'L')
    for at in mod_atoms_list:
        at.set_oniom_layer('L')
        at.set_new_index(at.get_index())

# ascribe atom names from info read from the pdb file:
    for at, pdb_at in zip(mod_atoms_list, at_list_from_pdb):
        at.set_name(pdb_at.get_name())

# determine and ascribe chain atribute
    chains = []
    gen_label = generate_label().__next__
    
    chain_indx = 0
    new_chain = peptide(gen_label(), chain_indx)
    
    for res in residues:
        if N_CO_in_residue(res):
            res.set_in_protein(True)
            chain_last_resid = new_chain.get_last_residue()
            if chain_last_resid:
                if is_peptide_bond2(chain_last_resid,res):
                    new_chain.add_residue(res)
                else:
                    chains.append(new_chain)
                    chain_indx += 1
                    new_chain = peptide(gen_label(), chain_indx)
                    new_chain.add_residue(res)
            else:
                new_chain.add_residue(res)
        elif new_chain.get_last_residue():
            chains.append(new_chain)
            chain_indx += 1
            new_chain = peptide(gen_label(), chain_indx) 

# set chain attribute for all residues belonging to peptide chains:
    for chain in chains:
        chain_label = chain.get_label()
        for resid in chain.get_residues():
            resid.set_chain(chain_label)
            
# set chain attribute to all other residues:        
    for resid in residues:
        if resid.get_chain() == '':
            resid.set_chain( gen_label() ) 


#   read info about H_layer and ascribe it to atoms
    Hl_res_ix, Hl_schain_ix, Hl_ix = read_rsi_index(input_f, "%H_layer", "%end_H_layer")
    mod_layer(residues, mod_atoms_list, Hl_res_ix, Hl_schain_ix, Hl_ix, 'H')
    
#   read info about M_layer
    Ml_res_ix, Ml_schain_ix, Ml_ix = read_rsi_index(input_f, "%M_layer", "%end_M_layer")
    mod_layer(residues, mod_atoms_list, Ml_res_ix, Ml_schain_ix, Ml_ix, 'M')

#   read info about H_L_link_atoms and generate a list of link_atom objects
#   H-link atom manipulations (HLA position, bonded_to)
    new_HL_lk_atoms = input_read_link_atoms(input_f, mod_atoms_list, border="HL")
    lk_atoms_mod(new_HL_lk_atoms, mod_atoms_list, 'H')
    
#   read info about H_M_link_atoms
#   H-link atom manipulations (HLA position, bonded_to)
    new_HM_lk_atoms = input_read_link_atoms(input_f, mod_atoms_list, border="HM")
    lk_atoms_mod(new_HM_lk_atoms, mod_atoms_list, 'H')

#   read info about M_L_link_atoms
#   H-link atom manipulations (HLA position, bonded_to)
    new_ML_lk_atoms = input_read_link_atoms(input_f, mod_atoms_list, border="ML")
    lk_atoms_mod(new_ML_lk_atoms, mod_atoms_list, 'M')

#   read info about NEW freeze_ref / r_free
    new_freeze = input_read_freeze(input_f, residues, mod_atoms_list)
    
    input_f.close()


    link_atoms_list = new_HL_lk_atoms + new_HM_lk_atoms + new_ML_lk_atoms   
#   check if all cut bonds are saturated with link atoms
    lk_at_indexes = []
    for at in link_atoms_list:
        lk_at_indexes.append(at.get_index())
        
    link_atoms_list.sort(key=lambda x: x.get_index(), reverse=False)
    lk_at_indexes.sort()    

    Hl_atoms = []
    Hl_indexes = []
    Ml_atoms = []
    Ml_indexes = []
    for at in mod_atoms_list:
        layer = at.get_oniom_layer()
        if layer == 'H':
            Hl_atoms.append(at)
            Hl_indexes.append( at.get_index() )
        elif layer == 'M':
            Ml_atoms.append(at)
            Ml_indexes.append( at.get_index() )
        
    link_atoms_updated = False    
    for at in Hl_atoms:
        qm_at_connect = at.get_connect_list()
        for item in qm_at_connect:
            if (item not in Hl_indexes) and (item not in lk_at_indexes.copy()):
                print("\nFound a QM-MM bond not capped with H-link atom")
                print("between atoms with (0-based) index of: ", at.get_index(), item)
                print("Adding a standard H-link atom with HC type\n")
                new_lk_atom = atom_to_link_atom(mod_atoms_list[item], 'HC', 0.000001, layer = 'H')
                new_lk_atom.set_bonded_to(at.get_index())
                new_lk_atom.set_new_type( 'HC' )
                adjust_HLA_coords(new_lk_atom, at)
                link_atoms_list.append(new_lk_atom)
                lk_at_indexes.append(item)
                link_atoms_updated = True

    for at in Ml_atoms:
        qm_at_connect = at.get_connect_list()
        for item in qm_at_connect:
            if (item not in Ml_indexes) and (item not in lk_at_indexes.copy()):
                print("\nFound a QM-MM bond not capped with H-link atom")
                print("between atoms with (0-based) index of: ", at.get_index(), item)
                print("Adding a standard H-link atom with HC type\n")
                new_lk_atom = atom_to_link_atom(mod_atoms_list[item], 'HC', 0.000001, layer = 'M')
                new_lk_atom.set_bonded_to(at.get_index())
                new_lk_atom.set_new_type( 'HC' )
                adjust_HLA_coords(new_lk_atom, at)
                link_atoms_list.append(new_lk_atom)
                lk_at_indexes.append(item)
                link_atoms_updated = True
    
    # sort link_atoms if this list was expanded:
    if link_atoms_updated:
        link_atoms_list.sort(key=lambda x: x.get_index(), reverse=False)
        lk_at_indexes.sort()

    n_at_in_oniom, n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer,\
        n_link_atoms_for_H, n_link_atoms_for_M = count_atoms_in_layers(mod_atoms_list, link_atoms_list)
    nlayers = nlayers_ONIOM(n_atom_in_H_layer, n_atom_in_M_layer, n_atom_in_L_layer)
    str_layers = ''
    if n_atom_in_L_layer > 0:
        str_layers += 'L, '
    if n_atom_in_M_layer > 0:
        str_layers += 'M, '
    if n_atom_in_H_layer > 0:
        str_layers += 'H '        
    print("\nThe ONIOM system to be written has: ", nlayers, " layers: ", str_layers)
    
    
#   inform about charges in the to be written ONIOM file
    sum_charges = charge_summary(mod_atoms_list, link_atoms_list, inp_p_charges)
    print("\nTotal charges in the output ONIOM input file:")
    report_charges(sum_charges)

    print("\nCharge and multiplicity for the ONIOM subsystems are: ")
    print("\nModel (H + link atoms):        ", qH, mH)
    print("\nIntermediate (M + link atoms): ", qM, mM)
    print("\nReal:                          ", qL, mH)
 
    mod_charge_and_spin = deepcopy(inp_charge_and_spin)
    if qH:
        mod_charge_and_spin["ChrgModelHigh"] = qH
        mod_charge_and_spin["ChrgModelMed"] = qH
        mod_charge_and_spin["ChrgModelLow"] = qH
    if mH:
        mod_charge_and_spin["SpinModelHigh"] = mH
        mod_charge_and_spin["SpinModelMed"] = mH
        mod_charge_and_spin["SpinModelLow"] = mH
    if qM:
        mod_charge_and_spin["ChrgIntMed"] = qM
        mod_charge_and_spin["ChrgIntLow"] = qM
    if mM:
        mod_charge_and_spin["SpinIntlMed"] = mH
        mod_charge_and_spin["SpinIntLow"] = mH
    if qL:
        mod_charge_and_spin["ChrgRealLow"] = qL
    if mL:
        mod_charge_and_spin["SpinRealLow"] = mL

#   write ouput files
    comment = inp_comment + "modifications read from: " + add_inp_fname + "\n"
    out_file = open(output_fname, 'a')
    write_oniom_inp_file(out_file, inp_header, comment, mod_charge_and_spin, nlayers,\
                         mod_atoms_list, link_atoms_list, inp_connect, inp_redundant, inp_params, inp_p_charges)

    out_file.close()

    pdb_out_fname = output_fname[0:-3] + 'MODEL.pdb'
    print("\nWriting a pdb file with the new model to the file: ", pdb_out_fname)
    write_pdb_file(residues, pdb_out_fname, write_Q=False)    
 
