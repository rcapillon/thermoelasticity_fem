import meshio


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
        self.n_dofs = self.n_nodes * 3

        for i, arr in enumerate(mesh.cell_data['gmsh:physical']):
            tag = arr[0]
            if mesh.cells[i]['type'] == 'vertex':
                self.dict_nodes_groups[tag] = mesh.cells[i].data.flatten()
            elif mesh.cells[i]['type'] == 'triangle':
                self.dict_tri_groups[tag] = mesh.cells[i].data
            elif mesh.cells[i]['type'] == 'tetra':
                self.dict_tet_groups[tag] = mesh.cells[i].data

    def set_materials(self, dict_materials):
        self.dict_materials = dict_materials

    def make_elements(self):
        """"""
