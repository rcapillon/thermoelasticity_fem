import meshio
import numpy as np

from thermoelasticity_fem.elements import Element


class Mesh:
    def __init__(self):
        self.path_to_mesh = None
        self.n_nodes = None
        self.n_dofs = None
        self.n_elements = None
        self.table_nodes = None
        self.dict_materials = {}
        self.dict_nodes_groups = {}
        self.dict_tri_groups = {}
        self.dict_tet_groups = {}
        self.elements = []

    def load_mesh(self, path_to_mesh):
        self.path_to_mesh = path_to_mesh
        mesh = meshio.read(path_to_mesh)
        self.table_nodes = mesh.points
        self.n_nodes = self.table_nodes.shape[0]
        self.n_dofs = self.n_nodes * 4

        for i, arr in enumerate(mesh.cell_data['gmsh:physical']):
            tag = arr[0]
            if mesh.cells[i].type == 'vertex':
                if tag not in self.dict_nodes_groups.keys():
                    self.dict_nodes_groups[tag] = mesh.cells[i].data.flatten()
                else:
                    self.dict_nodes_groups[tag] = np.concatenate((self.dict_nodes_groups[tag],
                                                                  mesh.cells[i].data.flatten()))
            elif mesh.cells[i].type == 'triangle':
                if tag not in self.dict_tri_groups.keys():
                    self.dict_tri_groups[tag] = mesh.cells[i].data
                else:
                    self.dict_tri_groups[tag] = np.concatenate((self.dict_tri_groups[tag], mesh.cells[i].data), axis=0)
            elif mesh.cells[i].type == 'tetra':
                if tag not in self.dict_tet_groups.keys():
                    self.dict_tet_groups[tag] = mesh.cells[i].data
                else:
                    self.dict_tet_groups[tag] = np.concatenate((self.dict_tet_groups[tag], mesh.cells[i].data), axis=0)

    def set_materials(self, dict_materials):
        self.dict_materials = dict_materials

    def make_elements(self):
        element_num = 0
        for tag, table_tets in self.dict_tet_groups.items():
            material = self.dict_materials[tag]
            for i in range(table_tets.shape[0]):
                nodes_nums = table_tets[i, :]
                nodes_coords = self.table_nodes[nodes_nums, :]
                element = Element(element_num, material, nodes_nums, nodes_coords)
                self.elements.append(element)
                element_num += 1

    def make_meshio_mesh(self):
        tets = []
        for v in self.dict_tet_groups.values():
            tets.extend(v.tolist())
        cells = [('tetra', tets)]
        mesh = meshio.Mesh(self.table_nodes, cells)

        return mesh

