from memdof import ExtendedTopologyInfo
from library.datagen.topology import load_extended_topology_info

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from library.config import Keys, config

START = 29

def rotation_matrix_axis(axis, angle):
    # Normalize the axis
    axis = axis / la.norm(axis)
    
    # Get the components
    x = axis[0]
    y = axis[1]
    z = axis[2]
    
    # Get the angle
    c = np.cos(angle)
    s = np.sin(angle)
    
    # Calculate the matrix
    R = np.array([
        [c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y**2 * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c)]
    ])
    
    return R


class Bond:
    def __init__(self, i, j, value):
        self.i = i - START
        self.j = j - START
        self.value = value

    def __str__(self) -> str:
        return f"B<{self.i}->{self.j}: {self.value}>"


class Angle:
    def __init__(self, i, j, k, value):
        self.i = i - START
        self.j = j - START
        self.k = k - START
        self.value = np.deg2rad(value)

    def __str__(self) -> str:
        return f"A<{self.i}-{self.j}-{self.k}: {np.rad2deg(self.value)}>"


class Dihedral:
    def __init__(self, i, j, k, l, value):
        self.i = i - START
        self.j = j - START
        self.k = k - START
        self.l = l - START
        self.value = np.deg2rad(value)

    def __str__(self) -> str:
        return f"D<{self.i}-{self.j}-{self.k}-{self.l}: {np.rad2deg(self.value)}>"

def plot_positions(pos: list, filename: str, range: list = None, bonds: list = None, angles: list = None, dihedrals: list = None):
    # Plot 3d positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(pos)
    pos = np.array([np.array(x) for x in pos])

    # Get the positions
    x = [p[0] for p in pos]
    y = [p[1] for p in pos]
    z = [p[2] for p in pos]

    # Plot labels
    for i, _ in enumerate(pos):
        ax.text(x[i], y[i], z[i], f"{i+START}", color="black", fontsize=6)

    print(pos)
    # Plot the positions
    ax.scatter(x, y, z, c='r', marker='o', edgecolors='k')

    # Plot the bonds
    if bonds:
        for bond in bonds:
            i = bond.i
            j = bond.j
            if i < len(pos) and j < len(pos):
                # Plot length
                bond_length = la.norm(pos[i] - pos[j])
                color = (
                    "green" if f"{bond_length:.2f}" == f"{bond.value:.2f}" else "red"
                )
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c=color)
                ax.text((x[i] + x[j]) / 2, (y[i] + y[j]) / 2, (z[i] + z[j]) / 2, f"{bond_length:.2f}", color=color, fontsize=6)

    # Plot the angles
    if angles:
        for angle in angles:
            i = angle.i
            j = angle.j
            k = angle.k
            if i < len(pos) and j < len(pos) and k < len(pos):
                # Calculate the angle
                a = pos[i] - pos[j]
                b = pos[k] - pos[j]
                angle_pred = np.arccos(np.dot(a, b) / (la.norm(a) * la.norm(b)))
                if angle_pred <= np.pi / 2:
                    angle_pred = np.pi - angle_pred
                print(np.rad2deg(angle_pred), np.rad2deg(angle.value))
                print(angle_pred == angle.value)
                color = "blue" if angle_pred == angle.value else "purple"

                # Highlight 10% of the bonds that consist of the angle (for better visibility)
                a = 0.4 * a / la.norm(a)
                b = 0.4 * b / la.norm(b)
                ax.plot([x[j], x[j] + a[0]], [y[j], y[j] + a[1]], [z[j], z[j] + a[2]], c=color)
                ax.plot([x[j], x[j] + b[0]], [y[j], y[j] + b[1]], [z[j], z[j] + b[2]], c=color)

                # Write the angle
                a = 0.1 * a / la.norm(a)
                b = 0.1 * b / la.norm(b)
                ax.text(
                    x[j] + a[0] + b[0],
                    y[j] + a[1] + b[1],
                    z[j] + a[2] + b[2],
                    f"{np.rad2deg(angle_pred):.2f}",
                    color=color,
                    fontsize=6,
                )

    if dihedrals:
        for dihedral in dihedrals:
            i = dihedral.i
            j = dihedral.j
            k = dihedral.k
            l = dihedral.l
            if i < len(pos) and j < len(pos) and k < len(pos) and l < len(pos):
                # Calculate the dihedral
                a = pos[i] - pos[j]
                b = pos[k] - pos[j]
                c = pos[l] - pos[k]
                plane1 = np.cross(a, b)
                plane2 = np.cross(b, c)
                angle_pred = np.arccos(np.dot(plane1, plane2) / (la.norm(plane1) * la.norm(plane2)))
                if angle_pred <= np.pi / 2:
                    angle_pred = np.pi - angle_pred
                print(np.rad2deg(angle_pred), np.rad2deg(dihedral.value))
                print(angle_pred == dihedral.value)
                color = "green" if angle_pred == dihedral.value else "orange"

                # Plot the two normal vectors and the dihedral angle
                ax.plot(
                    [x[k], x[k] + 0.2 * plane2[0] / la.norm(plane1)],
                    [y[k], y[k] + 0.2 * plane2[1] / la.norm(plane1)],
                    [z[k], z[k] + 0.2 * plane2[2] / la.norm(plane1)],
                    c=color,
                )
                ax.plot(
                    [x[k], x[k] + 0.2 * plane1[0] / la.norm(plane1)],
                    [y[k], y[k] + 0.2 * plane1[1] / la.norm(plane1)],
                    [z[k], z[k] + 0.2 * plane1[2] / la.norm(plane1)],
                )

                # Write the dihedral angle
    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if range:
        ax.set_xlim(range[0], range[1])
        ax.set_ylim(range[0], range[1])
        ax.set_zlim(range[0], range[1])

    # Make scale equal
    ax.set_box_aspect([1,1,1])

    # Save the plot
    plt.savefig(filename)
    plt.show()
    plt.close()

topo: ExtendedTopologyInfo = load_extended_topology_info()
atoms = [a for a in topo.atoms if "H" not in a["atom"]]
bonds = [b for b in topo.bonds]
angles = [a for a in topo.angles]
dihedrals = [d for d in topo.dihedrals]


# The indices of the atoms are relative to the cgnr so we will reindex them
for i, atom in enumerate(atoms):
    atom["nr"] = i

for k, bond in enumerate(bonds):
    # Translate i,j (cgnr) -> (index nr)
    i = bond["i"]
    j = bond["j"]
    
    atom_i = [a for a in atoms if a["cgnr"] == i][0]
    atom_j = [a for a in atoms if a["cgnr"] == j][0]
    
    i = atom_i["nr"]
    j = atom_j["nr"]
    
    bond["i"] = i
    bond["j"] = j


for x, angle in enumerate(angles):
    # Translate i,j,k (cgnr) -> (index nr)
    i = angle["i"]
    j = angle["j"]
    k = angle["k"]
    
    atom_i = [a for a in atoms if a["cgnr"] == i][0]
    atom_j = [a for a in atoms if a["cgnr"] == j][0]
    atom_k = [a for a in atoms if a["cgnr"] == k][0]
    
    i = atom_i["nr"]
    j = atom_j["nr"]
    k = atom_k["nr"]
    
    angle["i"] = i
    angle["j"] = j
    angle["k"] = k


for x, dihedral in enumerate(dihedrals):
    # Translate i,j,k,l (cgnr) -> (index nr)
    i = dihedral["i"]
    j = dihedral["j"]
    k = dihedral["k"]
    l = dihedral["l"]
    
    atom_i = [a for a in atoms if a["cgnr"] == i][0]
    atom_j = [a for a in atoms if a["cgnr"] == j][0]
    atom_k = [a for a in atoms if a["cgnr"] == k][0]
    atom_l = [a for a in atoms if a["cgnr"] == l][0]
    
    i = atom_i["nr"]
    j = atom_j["nr"]
    k = atom_k["nr"]
    l = atom_l["nr"]
    
    dihedral["i"] = i
    dihedral["j"] = j
    dihedral["k"] = k
    dihedral["l"] = l


# Make objects
bonds = [Bond(b["i"], b["j"], b["mean"]) for b in bonds]
angles = [Angle(a["i"], a["j"], a["k"], a["mean"]) for a in angles]
dihedrals = [Dihedral(d["i"], d["j"], d["k"], d["l"], d["mean"]) for d in dihedrals]

atoms = [a for a in atoms if a["nr"] >= START]
bonds = [b for b in bonds if min(b.i, b.j) >= 0]
angles = [a for a in angles if min(a.i, a.j, a.k) >= 0]
dihedrals = [d for d in dihedrals if min(d.i, d.j, d.k, d.l) >= 0]

[print(b) for b in bonds]
[print(b) for b in angles]
[print(b) for b in dihedrals]

print(len(atoms), len(bonds), len(angles), len(dihedrals))


# Calcualte the positions of the backbone starting at nr 30
pos = np.empty((len(atoms), 3))

# Set the first atom to the origin
pos[0] = [0, 0, 0]

# Set the second atom to the x-axis
pos[1] = [bonds[0].value, 0, 0]

# Set the thirs atom to the xy-plane
pos[2] = [
    pos[1][0] + bonds[1].value * np.cos(angles[0].value), 
    pos[1][1] + bonds[1].value * np.sin(angles[0].value),
    0
]

# Set the fourth atom
# This is done by calculating the rel_positiong to the local coordinate system of pos 3
# and then rotating it around the axis of the bond between pos 2 and pos 3 to get the dihedral right
# we just need to transform the local coordinate system of pos 3 to the global coordinate system
# by rotating it around the axis of the normal vector of the previous plane

# Step 1
rel_pos = [
    bonds[2].value * np.cos(angles[1].value), 
    bonds[2].value * np.sin(angles[1].value),
    0
]

# Translate the relative position to the absolute position by rotating around normal vector of the previous plane
normal_vector = np.cross(pos[1] - pos[0], pos[2] - pos[1])
rel_pos = np.dot(rotation_matrix_axis(normal_vector, -angles[0].value), rel_pos)

# rotate the fourth atom around the axis of the bond between pos 2 and pos 3
axis = pos[2] - pos[1]
rel_pos = np.dot(rotation_matrix_axis(axis, dihedrals[0].value), rel_pos)

# Switch x and y
rel_pos = [rel_pos[1], rel_pos[0], rel_pos[2]]


pos[3] = pos[2] + rel_pos

pos = [
    [0.0, 0.0, 0.0],
    [1.509, 0.0, 0.0],
    [0.08747740714692553, -0.0869885503382013, -0.5963577031600507],
    [0.26610356292136783, 0.24815076523575136, -2.089828605330517],
    [-0.23602546085104092, -1.194605193907242, -2.3140108656517775],
    [-1.5438329048931836, -0.8334971525872401, -3.048918799887073],
    [-1.0056380752103586, -1.4802759448755403, -4.332760113737333],
    [-2.0196352380594513, -2.6298648233803363, -4.314208452030723],
    [-2.702124322280674, -2.1191567315343187, -5.582344838560925],
    [-2.18103071940811, -3.397753930674363, -6.273421288309087],
    [-3.5622588238269657, -4.031716238553155, -6.496124489346906],
    [-3.3324363870994604, -3.8758933393702857, -8.008852115744626],
    [-3.282660807976059, -5.400647407941092, -8.204022354509922],
    [-4.580861676102703, -5.065546923777147, -8.967718086793608],
    [-4.309967399739276, -5.7149586676437965, -10.30035820248937],
    [-5.492640453134242, -6.352615495630583, -10.239504127069095],
    [-6.024254013181466, -5.537031486735281, -11.392452845446536],
    [-6.1323295346593705, -6.786137075910907, -12.295270605987108],
    [-7.670193544394653, -6.7116245988302055, -12.327242129570081],
    [-7.713971617688755, -6.423397593292992, -13.840414186402253],
    [-8.37879790872162, -7.776412918350611, -14.154986170073256],
    [-7.723363957507206, -6.406210336678214, -13.988447223355804],
]


n = 20
plot_positions(pos, "test.png", bonds=bonds, angles=angles, dihedrals=dihedrals)
