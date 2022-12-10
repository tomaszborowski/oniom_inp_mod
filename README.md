# oniom_inp_mod
python3 script to read and modify Gaussian ONIOM input files

Command line arguments:
    1. file name of an oniom input to read 
    2. name of the file to be produced (of type: oniom input, qm input, xyz)
    3. string switch, one from the list given below
    4. expected only for the switches: "rag", "rqg", "rqq" and "omod" - name of an additional input file 
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
