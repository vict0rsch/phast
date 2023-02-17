import os

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Embedding, Linear

os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"
from mendeleev.fetch import fetch_ionization_energies, fetch_table


class PhysRef(nn.Module):

    default_properties = [
        "atomic_radius",
        "atomic_volume",
        "density",
        "dipole_polarizability",
        "electron_affinity",
        "en_allen",
        "vdw_radius",
        "metallic_radius",
        "metallic_radius_c12",
        "covalent_radius_pyykko_double",
        "covalent_radius_pyykko_triple",
        "covalent_radius_pyykko",
        "IE1",
        "IE2",
    ]

    def __init__(
        self,
        properties=True,
        properties_grad=False,
        period=True,
        group=True,
        short=False,
        n_elements=85,
    ) -> None:
        """
        Create physical embeddings meta class with sub-emeddings for each atom

        Args:
            properties (bool, optional): Whether to create an embedding of physical
                embeddings. Defaults to True.
            properties_grad (bool, optional): Whether the physical properties embedding
                should be learned or kept fixed. Defaults to False.
            period (bool, optional): Whether to use period embeddings.
                Defaults to False.
            group (bool, optional): Whether to use group embeddings.
                Defaults to False.
            short (bool, optional)
            n_elements (int, optional): Number of elements to consider. Defaults to 85.
        """
        super().__init__()

        self.properties_list = [
            "atomic_radius",
            "atomic_volume",
            "density",
            "dipole_polarizability",
            "electron_affinity",
            "en_allen",
            "vdw_radius",
            "metallic_radius",
            "metallic_radius_c12",
            "covalent_radius_pyykko_double",
            "covalent_radius_pyykko_triple",
            "covalent_radius_pyykko",
            "IE1",
            "IE2",
        ]
        self.n_groups = 0
        self.n_periods = 0
        self.n_properties = 0

        self.properties = properties
        self.properties_grad = properties_grad
        self.period = period
        self.group = group
        self.short = short

        # Load table with all properties of all periodic table elements
        df = fetch_table("elements")
        df = df.set_index("atomic_number")

        # Add ionization energy
        ies = fetch_ionization_energies(degree=[1, 2])
        df = pd.concat([df, ies], axis=1)

        # Fetch group and period data
        if self.group:
            df.group_id = df.group_id.fillna(value=19.0)
            self.n_groups = int(df.group_id.loc[:n_elements].max() + 1)
            group_mapping = torch.cat(
                [torch.ones(1), torch.tensor(df.group_id.loc[:n_elements].values)]
            ).long()
            self.register_buffer("group_mapping", group_mapping)

        if self.period:
            self.n_periods = int(df.period.loc[:n_elements].max() + 1)
            period_mapping = torch.cat(
                [torch.ones(1), torch.tensor(df.period.loc[:n_elements].values)]
            ).long()
            self.register_buffer("period_mapping", period_mapping)

        if self.properties:
            # Create an embedding of physical properties
            # Select only potentially relevant elements
            df = df[self.properties_list]
            df = df.loc[:n_elements, :]

            # ! Normalize TODO: document this
            df = (df - df.mean()) / df.std()

            # Process 'NaN' values and remove further non-essential columns
            # ! Normalize TODO: document this
            if self.short:
                self.properties_list = df.columns[~df.isnull().any()].tolist()
                df = df[self.properties_list]
            else:
                self.properties_list = df.columns[
                    pd.isnull(df).sum() < int(1 / 2 * df.shape[0])
                ].tolist()
                df = df[self.properties_list]
                col_missing_val = df.columns[df.isna().any()].tolist()
                df[col_missing_val] = df[col_missing_val].fillna(
                    value=df[col_missing_val].mean()
                )

            self.n_properties = len(df.columns)
            properties_mapping = torch.cat(
                [
                    torch.zeros(1, self.n_properties),
                    torch.from_numpy(df.values).float(),
                ]
            )
            self.register_buffer("properties_mapping", properties_mapping)

    def __repr__(self):
        return f"PhysRef(properties={self.properties}, properties_grad={self.properties_grad}, period={self.period}, group={self.group}, short={self.short})"  # noqa: E501

    def period_and_group(self, z):
        values = {}
        if self.period:
            values["period"] = self.period_mapping[z]
        if self.group:
            values["group"] = self.group_mapping[z]
        return values


class PropertiesEmbedding(nn.Module):
    def __init__(self, properties, grad=False):
        super().__init__()
        assert isinstance(properties, torch.Tensor)
        assert isinstance(grad, bool)

        if grad:
            self.register_parameter("properties", nn.Parameter(properties))
        else:
            self.register_buffer("properties", properties)

    def forward(self, z):
        return self.properties[z]

    def reset_parameters(self):
        pass


class PhysEmbedding(nn.Module):
    def __init__(
        self,
        z_emb_size=32,
        tag_emb_size=32,
        period_emb_size=32,
        group_emb_size=32,
        properties=PhysRef.default_properties,
        properties_proj_size=0,
        properties_grad=False,
        final_proj_size=0,
        n_elements=85,
    ):
        super().__init__()
        self.z_emb_size = z_emb_size
        self.tag_emb_size = tag_emb_size
        self.period_emb_size = period_emb_size
        self.group_emb_size = group_emb_size
        self.properties_proj_size = properties_proj_size
        self.final_proj_size = final_proj_size
        self.properties = properties
        self.properties_grad = properties_grad
        self.n_elements = n_elements

        self.phys_lin = None
        self.z_emb = None
        self.period_emb = None
        self.group_emb = None
        self.group_emb = None
        self.final_proj = None

        # Check phys_emb_type is valid
        assert properties_grad in {
            True,
            False,
        }, f"Unknown properties_grad {properties_grad}. Allowed: True or False."

        if self.properties_proj_size > 0 and not self.properties:
            raise ValueError(
                "Cannot project physical properties if `self.properties` is empty."
            )

        # Check embedding sizes are non-negative
        for emb_name, emb_size in {
            "z_emb_size": z_emb_size,
            "tag_emb_size": tag_emb_size,
            "period_emb_size": period_emb_size,
            "group_emb_size": group_emb_size,
        }.items():
            assert (
                emb_size >= 0
            ), f"Embedding size must be non-negative, got {emb_size} for {emb_name}"

        self.full_emb_size = int(
            self.z_emb_size
            + self.tag_emb_size
            + self.period_emb_size
            + self.group_emb_size
            + self.properties_proj_size * int(bool(self.properties))
        )

        # Physical properties
        self.phys_ref = PhysRef(
            properties=self.properties,
            properties_grad=self.properties_grad,
            period=self.period_emb_size > 0,
            group=self.group_emb_size > 0,
            n_elements=n_elements,
        )

        self.embeddings = nn.ModuleDict()

        # Main embedding
        if self.z_emb_size > 0:
            self.embeddings["z"] = Embedding(n_elements, self.z_emb_size)

        # With projection?
        if self.properties:
            properties_embedding = PropertiesEmbedding(
                self.phys_ref.properties_mapping, self.properties_grad
            )
            if self.properties_proj_size > 0:
                self.phys_lin = Linear(self.phys_ref.n_properties, properties_proj_size)
                self.embeddings["properties"] = nn.Sequential(
                    properties_embedding, self.phys_lin
                )
            else:
                self.embeddings["properties"] = properties_embedding
                self.full_emb_size += self.phys_ref.n_properties

        if self.full_emb_size == 0:
            raise ValueError("Total embedding size is 0!")

        # Period embedding
        if self.period_emb_size > 0:
            self.embeddings["period"] = Embedding(
                self.phys_ref.n_periods, self.period_emb_size
            )

        # Group embedding
        if self.group_emb_size > 0:
            self.embeddings["group"] = Embedding(
                self.phys_ref.n_groups, self.group_emb_size
            )

        # Tag embedding
        if self.tag_emb_size > 0:
            self.embeddings["tag"] = Embedding(3, self.tag_emb_size)

        if self.final_proj_size > 0:
            self.final_proj = Linear(self.full_emb_size, self.final_proj_size)

        self.reset_parameters()

    def reset_parameters(self):
        if self.phys_lin:
            nn.init.xavier_uniform_(self.phys_lin.weight)
        for emb in self.embeddings.values():
            if isinstance(emb, (nn.Sequential, PhysRef)):
                pass
            else:
                emb.reset_parameters()

    def forward(self, z, tag=None):
        pg = self.phys_ref.period_and_group(z.long())
        h = []

        for e, emb in self.embeddings.items():
            if e in pg:
                h.append(emb(pg[e]))
            elif e in {"z", "properties"}:
                h.append(emb(z))
            elif e == "tag":
                assert tag is not None, "Tag embedding is used but no tag is provided."
                h.append(emb(tag))

        h = torch.cat(h, dim=-1)

        if self.final_proj:
            h = self.final_proj(h)

        return h
