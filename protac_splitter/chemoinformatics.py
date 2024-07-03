import random
from typing import List, Tuple, Callable, Any, Union, Dict, Optional, Literal

import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    rdFingerprintGenerator,
    rdMolHash,
    rdFMCS,
    rdMolAlign,
)


def standardize_smiles(smiles: str) -> str:
    """
    Standardizes a given SMILES string.

    Args:
        smiles (str): The input SMILES string to be standardized.

    Returns:
        str: The standardized SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        else:
            print(f'Smile returned error: {smiles}')
            return None
    except:
        print(f'Smile returned error: {smiles}')
        return None
        # raise ValueError(f'Failed to process smile: {smiles}')


def remove_stereo(smiles: str) -> str:
    """
    Remove stereochemistry from a SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The SMILES string with stereochemistry removed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.rdmolops.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except:
        return np.nan


def get_mol(smiles: str) -> Chem.Mol:
    """
    Get a molecule object from a SMILES string.

    Args:
        smiles (str): The SMILES string representing the molecule.

    Returns:
        Chem.Mol: The molecule object.
    """
    mol = Chem.MolFromSmiles(smiles)
    Chem.rdmolops.RemoveStereochemistry(mol)
    return mol


def compute_RDKitFP(
        smiles: Union[str, List[str], List[Chem.Mol]],
        maxPath: int = 7,
        fpSize: int = 2048,
) -> List[Chem.RDKFingerprint]:
    """
    Compute RDKit fingerprints for a given list of SMILES strings or RDKit molecules.

    Args:
        smiles (Union[str, List[str], List[Chem.Mol]]): A single SMILES string or a list of SMILES strings
            or a list of RDKit molecules.
        maxPath (int, optional): The maximum path length for the fingerprints. Defaults to 7.
        fpSize (int, optional): The size of the fingerprint vector. Defaults to 2048.

    Returns:
        List[Chem.RDKFingerprint]: A list of RDKit fingerprints computed from the input SMILES strings or molecules.
    """
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles  # assume mols were fed instead
    rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(
        maxPath=maxPath, fpSize=fpSize)
    fps = [rdgen.GetCountFingerprint(mol) for mol in mols]
    return fps


def compute_countMorgFP(
        smiles: List[str],
        radius: int = 2,
) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Compute the count-based Morgan fingerprint for a list of SMILES strings.

    Args:
        smiles (List[str]): A list of SMILES strings.
        radius (int, optional): The radius parameter for the Morgan fingerprint. Defaults to 2.

    Returns:
        List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]: A list of count-based Morgan fingerprints.
    """
    if smiles is None:
        return None
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles  # assume mols were fed instead
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    fps = [fpgen.GetCountFingerprint(mol) for mol in mols]
    return fps


def tanimoto_similarity_matrix(fps, return_distance=False):
    """
    Calculate a symmetric Tanimoto similarity matrix for a list of fingerprints using bulk operations.

    Parameters:
    - fps: list, RDKit fingerprint objects for which to calculate similarity.

    Returns:
    - np.array, Symmetric square matrix of Tanimoto similarity.
    """
    num_fps = len(fps)
    # Initialize a square matrix of zeros
    sim_matrix = np.zeros((num_fps, num_fps))

    for i in tqdm(range(num_fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim = np.array(similarities)
        sim_matrix[i, :] = sim
        # Set diagonal to 1 as the similarity to self is 1
        sim_matrix[i, i] = 1

    if return_distance:
        return 1 - sim_matrix
    return sim_matrix


def add_attachments(
        list_without_attachments: List[str],
        dict_map_without_to_with: Dict[str, List[str]],
) -> Tuple[List[str], List[Chem.Mol]]:
    """
    Adds attachments to a list of molecules.

    Args:
        list_without_attachments (List[str]): A list of SMILES strings representing molecules without attachments.
        dict_map_without_to_with (Dict[str, List[str]]): A dictionary mapping SMILES strings without attachments to a list of SMILES strings with attachments.

    Returns:
        Tuple[List[str], List[Chem.Mol]]: A tuple containing two lists:
            - smiles_with_attachment (List[str]): A list of SMILES strings representing molecules with attachments.
            - mols (List[Chem.Mol]): A list of RDKit molecule objects corresponding to the molecules with attachments.
    """
    smiles_with_attachment = []
    for smi in list_without_attachments:
        smiles_with_attachment.extend(dict_map_without_to_with[smi])
    smiles_with_attachment = list(set(smiles_with_attachment))
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_with_attachment]
    return smiles_with_attachment, mols


def reassemble_protac(
        poi_smiles: str,
        linker_smiles: str,
        e3_smiles: str,
        bond_type: str,
) -> Tuple[str, Chem.Mol]:
    """
    Reassembles a PROTAC molecule by merging the POI, linker, and E3 substructures.

    Args:
        poi_smiles (str): The SMILES representation of the POI (Protein of Interest) substructure.
        linker_smiles (str): The SMILES representation of the linker substructure.
        e3_smiles (str): The SMILES representation of the E3 substructure.
        bond_type (str): The type of bond to be formed between the substructures.

    Returns:
        Tuple[str, Chem.Mol]: A tuple containing the SMILES representation and RDKit Mol object of the reassembled PROTAC molecule.
    """

    if "[*:1]" in e3_smiles:
        raise ValueError(f"[*:1] found among E3-SMILES: {e3_smiles}]")
    elif "[*:2]" in poi_smiles:
        raise ValueError(f"[*:2] found among POI-SMILES: {poi_smiles}]")
    elif "[*:1]" not in linker_smiles or "[*:2]" not in linker_smiles:
        raise ValueError(
            f"[*:1] or [*:2] missing among Linker-SMILES: {linker_smiles}]")

    # Convert SMILES to RDKit Molecule objects
    poi_mol = Chem.MolFromSmiles(poi_smiles)
    linker_mol = Chem.MolFromSmiles(linker_smiles)
    e3_mol = Chem.MolFromSmiles(e3_smiles)

    # Find the indices of the attachment points
    poi_l_attachment_points, _ = find_atom_index_of_mapped_atoms_detailed(
        poi_mol)
    linker_poi_attachment_points, linker_e3_attachment_points = find_atom_index_of_mapped_atoms_detailed(
        linker_mol)
    _, e3_l_attachment_points = find_atom_index_of_mapped_atoms_detailed(
        e3_mol)

    # Ensure that each molecule has the correct number of attachment points
    if not poi_l_attachment_points or not linker_poi_attachment_points or not linker_e3_attachment_points or not e3_l_attachment_points:
        raise ValueError(
            "Missing attachment points in one or more substructures")

    # Select the first (and only) attachment point for POI and E3, and the appropriate ones for the linker
    poi_idx = poi_l_attachment_points[0]
    linker_e3_idx = linker_e3_attachment_points[0]
    e3_idx = e3_l_attachment_points[0]

    # Merge E3 with Linker
    e3_linker_mol = merge_molecules(
        e3_mol, linker_mol, e3_idx, linker_e3_idx, bond_type=bond_type)
    linker_e3_mol_attachment_point, _ = find_atom_index_of_mapped_atoms_detailed(
        e3_linker_mol)
    linker_e3_mol_idx = linker_e3_mol_attachment_point[0]

    protac_mol = merge_molecules(
        e3_linker_mol, poi_mol, linker_e3_mol_idx, poi_idx, bond_type=bond_type)
    Chem.SanitizeMol(protac_mol)
    protac_smiles = Chem.MolToSmiles(protac_mol, canonical=True)

    return protac_smiles, protac_mol


def merge_molecules(
        mol1: Chem.Mol,
        mol2: Chem.Mol,
        atom_idx1: int,
        atom_idx2: int,
        bond_type: Literal['single', 'rand_uniform'] = 'single',
) -> Chem.Mol:
    """
    Merge two molecules into a single editable molecule.

    Args:
        mol1 (Chem.Mol): The first molecule.
        mol2 (Chem.Mol): The second molecule.
        atom_idx1 (int): The index of the attachment point in mol1.
        atom_idx2 (int): The index of the attachment point in mol2.
        bond_type (Literal['single', 'rand_uniform'], optional): The type of bond to be formed between the attachment points. Defaults to 'single'.

    Returns:
        Chem.Mol: The merged molecule.

    Raises:
        ValueError: If the index is out of range.

    """
    # Combine the two molecules into a single editable molecule
    combined_mol = Chem.CombineMols(mol1, mol2)
    editable_mol = Chem.EditableMol(combined_mol)

    # Find neighbors of the attachment points
    neighbor_atom_idx1 = [nbr.GetIdx() for nbr in mol1.GetAtomWithIdx(
        atom_idx1).GetNeighbors() if nbr.GetAtomicNum() > 1][0]
    neighbor_atom_idx2 = [nbr.GetIdx() + mol1.GetNumAtoms()
                          for nbr in mol2.GetAtomWithIdx(atom_idx2).GetNeighbors() if nbr.GetAtomicNum() > 1]

    if neighbor_atom_idx2 == []:  # if linker has no length
        smi_e3_linker_with_e3_attachment = Chem.MolToSmiles(
            mol1, canonical=True)
        smi_e3_linker_with_poi_attachment = smi_e3_linker_with_e3_attachment.replace(
            "[*:2]", "[*:1]")
        mol_e3_linker_with_poi_attachment = Chem.MolFromSmiles(
            smi_e3_linker_with_poi_attachment)
        return mol_e3_linker_with_poi_attachment
    else:
        neighbor_atom_idx2 = neighbor_atom_idx2[0]
        # raise ValueError("Index out of range?")

    # Add a bond between the neighboring atoms (ignoring the dummy atoms)

    if bond_type == 'single':
        editable_mol.AddBond(
            neighbor_atom_idx1, neighbor_atom_idx2, order=Chem.rdchem.BondType.SINGLE)
    elif bond_type == 'rand_uniform':
        neighbor_atom1 = mol1.GetAtomWithIdx(neighbor_atom_idx1)
        neighbor_atom2 = mol2.GetAtomWithIdx(
            neighbor_atom_idx2-mol1.GetNumAtoms())
        highest_allowed_bondorder_atom_idx1 = neighbor_atom1.GetTotalNumHs() + \
            1  # +1 for the attatchment point
        highest_allowed_bondorder_atom_idx2 = neighbor_atom2.GetTotalNumHs() + 1
        highest_allowed_bondorder = min(
            [highest_allowed_bondorder_atom_idx1, highest_allowed_bondorder_atom_idx2])
        possible_bonds = [Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
        allowed_bonds = possible_bonds[0:highest_allowed_bondorder]
        sampled_bond = random.sample(allowed_bonds, 1)[0]
        editable_mol.AddBond(neighbor_atom_idx1,
                             neighbor_atom_idx2, order=sampled_bond)

    # Calculate the adjusted index for the attachment point in mol2
    adjusted_atom_idx2 = atom_idx2 + mol1.GetNumAtoms()

    # Remove the dummy atoms - IMPORTANT: remove the atom with the higher index first!
    max_idx = max(atom_idx1, adjusted_atom_idx2)
    min_idx = min(atom_idx1, adjusted_atom_idx2)

    editable_mol.RemoveAtom(max_idx)
    editable_mol.RemoveAtom(min_idx)

    # Get the modified molecule
    modified_mol = editable_mol.GetMol()

    # Sanitize the molecule to ensure its chemical validity
    Chem.SanitizeMol(modified_mol)

    return modified_mol


def substructure_split_sort(substructure_smiles: str) -> Tuple[str, str, str]:
    """
    Splits a substructure SMILES string into three parts: poi_smile, linker_smile, and e3_smile.

    Args:
        substructure_smiles (str): The substructure SMILES string to be split.

    Returns:
        Tuple[str, str, str]: A tuple containing the poi_smile, linker_smile, and e3_smile.

    Raises:
        ValueError: If [*:1] and [*:2] are not found in any of the substructure SMILES.
    """
    if isinstance(substructure_smiles, str):
        substructure_smiles = substructure_smiles.split(".")
    for smile in substructure_smiles:
        if '[*:1]' in smile:
            if '[*:2]' in smile:
                linker_smile = smile
            else:
                poi_smile = smile
        elif '[*:2]' in smile:
            e3_smile = smile
        else:
            raise ValueError(
                f'[*:1] and [*:2] was not found in smile: {smile}')
    return poi_smile, linker_smile, e3_smile


def identify_bad_substructure_match(protac_smile: str, poi_smile: str, e3_smile: str) -> bool:
    """
    Identifies if the substructure match between the PROTAC and the POI and E3 ligands is bad.

    Args:
        protac_smile (str): The SMILES representation of the PROTAC molecule.
        poi_smile (str): The SMILES representation of the POI (Protein of Interest) molecule.
        e3_smile (str): The SMILES representation of the E3 ligand molecule.

    Returns:
        bool: True if the substructure match is bad, False otherwise.
    """
    protac_mol = Chem.MolFromSmiles(protac_smile)
    poi_mol = Chem.MolFromSmiles(poi_smile)
    e3_mol = Chem.MolFromSmiles(e3_smile)

    poi_mol = remove_dummy_atoms(poi_mol)
    e3_mol = remove_dummy_atoms(e3_mol)

    matches_poi = protac_mol.GetSubstructMatches(poi_mol)
    matches_e3 = protac_mol.GetSubstructMatches(e3_mol)

    if len(matches_poi) == 1 and len(matches_e3) == 1:
        return False
    elif len(matches_poi) == 2 and len(matches_e3) == 2:  # OBS Work in progress
        return True
    else:
        return True


def remove_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    """
    Removes dummy atoms from a molecule.

    Args:
        mol (Chem.Mol): The molecule from which dummy atoms should be removed.

    Returns:
        Chem.Mol: The modified molecule with dummy atoms removed.
    """
    atoms_to_remove = [atom.GetIdx()
                       for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    editable_mol = Chem.EditableMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):
        editable_mol.RemoveAtom(idx)
    return editable_mol.GetMol()


def get_boundary_bondtype(mol: Chem.Mol, bondtype_count: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Get the count of different bond types connected to dummy atoms in a molecule.

    Args:
        mol (Chem.Mol): The molecule to analyze.
        bondtype_count (Optional[Dict[str, int]]): A dictionary to store the count of different bond types.
            Defaults to None.

    Returns:
        Dict[str, int]: A dictionary containing the count of different bond types connected to dummy atoms.
            The keys are the bond types ('SINGLE', 'DOUBLE', 'TRIPLE') and the values are the corresponding counts.
    """
    if bondtype_count is None:
        bondtype_count = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0}
    bondtype_to_str = {Chem.rdchem.BondType.SINGLE: 'SINGLE',
                       Chem.rdchem.BondType.DOUBLE: 'DOUBLE',
                       Chem.rdchem.BondType.TRIPLE: 'TRIPLE'}

    # Find dummy atoms by symbol or index
    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms()
                   if atom.GetSymbol() == '*']
    # reverse to avoid index shifting issues
    for dummy_atom_idx in reversed(dummy_atoms):
        # identify the order of the bond
        atom = mol.GetAtomWithIdx(dummy_atom_idx)
        neighbors = atom.GetNeighbors()

        for neighbour_atom in neighbors:
            neighbour_idx = neighbour_atom.GetIdx()
            dummy_atom_bond = mol.GetBondBetweenAtoms(
                dummy_atom_idx, neighbour_idx)
            dummy_atom_bondtype = dummy_atom_bond.GetBondType()
            bondtype_count[bondtype_to_str[dummy_atom_bondtype]] += 1

    return bondtype_count


def remove_dummy_atom_from_mol(mol: Chem.Mol, output: str = "smiles") -> Union[str, Chem.Mol]:
    """
    Removes dummy atoms from a molecule and returns the modified molecule.

    Args:
        mol (Chem.Mol): The input molecule containing dummy atoms.
        output (str, optional): The output format. Defaults to "smiles".

    Returns:
        Union[str, Chem.Mol]: The modified molecule without dummy atoms. If output is "smiles", returns the SMILES string representation of the molecule. Otherwise, returns the modified molecule as a Chem.Mol object.
    """

    if Chem.MolToSmiles(mol) == "O=C(CCCCCCCCCC[*:1])[*:2]":
        pass

    # Find dummy atoms by symbol or index
    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms()
                   if atom.GetSymbol() == '*']

    hydrogen_atom = Chem.rdchem.Atom(1)
    bond_to_num_Hs_to_add = {Chem.rdchem.BondType.SINGLE: 1,
                             Chem.rdchem.BondType.DOUBLE: 2,
                             Chem.rdchem.BondType.TRIPLE: 3}

    # identify the order of the bond
    # add that many hydrogens to the other atom the dummy atom is connected to
    # remove the dummy atom
    # RemoveHs

    # for each dummy atom
    emol = Chem.RWMol(mol)
    # reverse to avoid index shifting issues
    for dummy_atom_idx in reversed(dummy_atoms):
        # identify the order of the bond
        atom = mol.GetAtomWithIdx(dummy_atom_idx)
        neighbors = atom.GetNeighbors()

        for neighbour_atom in neighbors:
            neighbour_idx = neighbour_atom.GetIdx()
            dummy_atom_bond = emol.GetBondBetweenAtoms(
                dummy_atom_idx, neighbour_idx)
            dummy_atom_bondtype = dummy_atom_bond.GetBondType()
            num_Hs_to_add = bond_to_num_Hs_to_add[dummy_atom_bondtype]
            for _ in range(num_Hs_to_add):
                hydrogen_atom_idx = emol.AddAtom(hydrogen_atom)
                emol.AddBond(neighbour_idx, hydrogen_atom_idx,
                             order=Chem.rdchem.BondType.SINGLE)

        emol.RemoveAtom(dummy_atom_idx)

    substruct_mol_without_attachment_extra_hydrogens = Chem.Mol(emol)
    substruct_mol_without_attachment = Chem.RemoveHs(
        substruct_mol_without_attachment_extra_hydrogens)

    Chem.GetSymmSSSR(substruct_mol_without_attachment)
    Chem.SanitizeMol(substruct_mol_without_attachment)

    substruct_mol_without_attachment = Chem.MolFromSmiles(Chem.MolToSmiles(
        substruct_mol_without_attachment))  # verify this can be done...

    dummy_atoms = [atom.GetIdx() for atom in substruct_mol_without_attachment.GetAtoms(
    ) if atom.GetSymbol() == '*']
    if dummy_atoms != []:
        print(f'SMILES: {Chem.MolToSmiles(mol)}')
        # display(mol)
        print("new mol:")
        # display(substruct_mol_without_attachment)
        raise ValueError("dummy atoms still present!")

    if output == "smiles":
        return Chem.MolToSmiles(substruct_mol_without_attachment, canonical=True)
    else:
        return substruct_mol_without_attachment


def remove_dummy_atom_from_smiles(smiles: str, output: str = "smiles") -> str:
    """
    Remove dummy atom from a SMILES string.

    Args:
        smiles (str): The input SMILES string.
        output (str, optional): The output format. Defaults to "smiles".

    Returns:
        str: The modified SMILES string without the dummy atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    return remove_dummy_atom_from_mol(mol, output)


def find_atom_index_of_mapped_atoms_detailed(mol: Chem.Mol) -> Tuple[List[int], List[int]]:
    """
    Find the atom indices of the mapped atoms in a molecule.

    Args:
        mol (Chem.Mol): The molecule to search for mapped atoms.

    Returns:
        tuple: A tuple containing two lists. The first list contains the atom indices of the 
        atoms with atom map number 1, and the second list contains the atom indices of the 
        atoms with atom map number 2.

    Raises:
        ValueError: If there are more than one attachment points for either atom map number 1 
        or atom map number 2.
    """
    poi_l_attachment_point = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1]
    e3_l_attachment_point = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]

    if len(poi_l_attachment_point) > 1 or len(e3_l_attachment_point) > 1:
        raise ValueError("Too many attachment points")

    return poi_l_attachment_point, e3_l_attachment_point


def linker_mol_to_ms(mol: Chem.Mol) -> Chem.Mol:
    """
    Converts a linker molecule to its Murcko Scaffold representation.

    Args:
        mol (Chem.Mol): The linker molecule to be converted.

    Returns:
        Chem.Mol: The Murcko Scaffold representation of the linker molecule.
    """

    if mol.GetNumAtoms() == 2:  # only [*:1] and [*:2]
        return mol

    # if "[*:1]" and "[*:2]" in smiles

    poi_l_attachment_point, e3_l_attachment_point = find_atom_index_of_mapped_atoms_detailed(
        mol)

    emol = Chem.EditableMol(mol)

    # add one single bond between the attachment points
    try:
        emol.AddBond(
            poi_l_attachment_point[0], e3_l_attachment_point[0], Chem.Chem.rdchem.BondType.SINGLE)
    except:
        # display(mol)
        print(f'poi_l_attachment_point:{poi_l_attachment_point}')
        print(f'e3_l_attachment_point:{e3_l_attachment_point}')
        print(Chem.MolToSmiles(mol, canonical=True))
        raise ValueError("Fail add bond")

    mol_circulized = emol.GetMol()
    try:
        # Sanitize the molecule
        # Finding rings and re-perceiving aromaticity
        Chem.GetSymmSSSR(mol_circulized)
        Chem.SanitizeMol(mol_circulized)
    except:
        raise ValueError("Fail GetSymmSSSR or SanitizeMol")

    # apply MS
    mol_circulized_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(
        mol_circulized)
    ms_poi_l_attachment_point, ms_e3_l_attachment_point = find_atom_index_of_mapped_atoms_detailed(
        mol_circulized_ms)
    # mol_circulized_ms.GetBondBetweenAtoms(ms_poi_l_attachment_point, ms_e3_l_attachment_point).SetBondType(Chem.Chem.rdchem.BondType.UNSPECIFIED)

    emol_circulized_ms = Chem.EditableMol(mol_circulized_ms)

    # remove the bond between the attachment points
    emol_circulized_ms.RemoveBond(
        ms_poi_l_attachment_point[0], ms_e3_l_attachment_point[0])

    mol_ms = emol_circulized_ms.GetMol()

    try:
        # Sanitize the molecule
        Chem.GetSymmSSSR(mol_ms)  # Finding rings and re-perceiving aromaticity
        Chem.SanitizeMol(mol_ms)
    except:
        raise ValueError("Fail GetSymmSSSR or SanitizeMol")

    return mol_ms


def get_anonymous_mol(mol: Chem.Mol) -> str:
    """
    Get the anonymous graph representation of a molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The input molecule.

    Returns:
        str: The anonymous graph representation of the molecule.

    Raises:
        ValueError: If there is an error processing the molecule.
    """
    try:
        return rdMolHash.MolHash(mol, rdMolHash.HashFunction.AnonymousGraph)
    except:
        raise ValueError(
            f"Error processing molecule with rdMolHash.HashFunction.AnonymousGraph")


def get_anonymous_murcko(smiles: str) -> str:
    """
    Get the anonymous Murcko scaffold for a given SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The SMILES string representation of the anonymous Murcko scaffold.
    """
    mol = Chem.MolFromSmiles(smiles)
    # Handle invalid SMILES strings
    if mol is None:
        raise ValueError("mol is None")

    smi_anon = get_anonymous_mol(mol)
    mol_anon = Chem.MolFromSmiles(smi_anon)

    if "[*:1]" in smiles and "[*:2]" in smiles:
        mol_anon_ms = linker_mol_to_ms(mol_anon)
    else:
        mol_anon_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol_anon)

    # smi_anon_ms = get_anonymous_mol(mol_ms)
    smi_anon_ms = Chem.MolToSmiles(mol_anon_ms, canonical=True)

    return smi_anon_ms


def get_murcko(smiles: str) -> str:
    """
    Get the Murcko scaffold for a given SMILES string.

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str: The Murcko scaffold SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES strings
        raise ValueError("mol is None")

    if "[*:1]" in smiles and "[*:2]" in smiles:  # is_linker = True
        mol_ms = linker_mol_to_ms(mol)
    else:
        mol_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)

    smi_ms = Chem.MolToSmiles(mol_ms, canonical=True)

    return smi_ms


def get_bond_idx(smi: str, bonds_start_end_atoms: List[List[int]]) -> List[int]:
    """
    Get the indices of bonds in a molecule that match the given start and end atom indices.

    Args:
        smi (str): The SMILES representation of the molecule.
        bonds_start_end_atoms (List[List[int]]): A list of lists containing the start and end atom indices of the bonds to search for.

    Returns:
        List[int]: A list of bond indices that match the given start and end atom indices.
    """
    mol = Chem.MolFromSmiles(smi)

    bond_indices = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if [begin_idx, end_idx] in bonds_start_end_atoms or [end_idx, begin_idx] in bonds_start_end_atoms:
            bond_indices.append(bond.GetIdx())
        elif (begin_idx, end_idx) in bonds_start_end_atoms or (end_idx, begin_idx) in bonds_start_end_atoms:
            bond_indices.append(bond.GetIdx())

    return bond_indices
