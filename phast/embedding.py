"""
A Python module that endows graph neural networks with physical priors
as part of the embeddings of atoms from their characteristic number.

This package contains the implementation of a set of classes that are used to
create atomic embeddings from physical properties of periodic table elements.

The physical embeddings are learned or kept fixed depending on the specific use-case.
The embeddings can also include information regarding the group and period of the
elements.

In the context of the Open Catalyst datasets, tag embeddings can also be used.

This implementation relies on
`Mendeleev <https://mendeleev.readthedocs.io/en/stable/data.html>`_ package to access
the physical properties of elements from the periodic table.

.. image:: https://raw.githubusercontent.com/vict0rsch/phast/main/examples/data/embedding.png
    :alt: graph rewiring
    :width: 600px

.. code-block:: python

    import torch
    from phast.embedding import PhysEmbedding

    z = torch.randint(1, 85, (3, 12)) # batch of 3 graphs with 12 atoms each
    phys_embedding = PhysEmbedding(
        z_emb_size=32, # default
        period_emb_size=32, # default
        group_emb_size=32, # default
        properties_proj_size=32, # default is 0 -> no learned projection
        n_elements=85, # default
    )
    h = phys_embedding(z) # h.shape = (3, 12, 128)

    tags = torch.randint(0, 3, (3, 12))
    phys_embedding = PhysEmbedding(
        tag_emb_size=32, # default is 0, this is OC20-specific
        final_proj_size=64, # default is 0, no projection, just the concat. of embeds.
    )

    h = phys_embedding(z, tags) # h.shape = (3, 12, 64)


"""

import os
from typing import Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Embedding, Linear

os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"
from mendeleev.fetch import fetch_ionization_energies, fetch_table


class PhysRef(nn.Module):
    """
    This class implements an interface to access physical properties, period and
    group ids of elements from the periodic table.

    Attributes:
        default_properties (:obj:`list`): A list of the default properties part of
            atom embeddings.
        properties_list (:obj:`list`): A list of the properties that are actually used for
            creating the embeddings.
        n_groups (:obj:`int`): The number of groups of the elements.
        n_periods (:obj:`int`): The number of periods of the elements.
        n_properties (:obj:`int`): The number of properties of the elements that are used to
            create the embeddings.
        properties (:obj:`bool`): Whether to create an embedding of physical embeddings.
        properties_grad (:obj:`bool`): Whether the physical properties embedding should be
            learned or kept fixed.
        period (:obj:`bool`): Whether to use period embeddings.
        group (:obj:`bool`): Whether to use group embeddings.
        short (:obj:`bool`): A boolean flag indicating whether to keep only the columns that
            do not have NaN values.
        group_mapping (:obj:`torch.Tensor`): A tensor containing the mapping from the element
            atomic number to the corresponding group embedding.
        period_mapping (:obj:`torch.Tensor`): A tensor containing the mapping from the element
            atomic number to the corresponding period embedding.
        properties_mapping (:obj:`torch.Tensor`): A tensor containing the mapping from the
            element atomic number to the corresponding physical properties embedding.

    Methods:
        __init__: Initializes the PhysRef class.
        __repr__: Returns a string representation of the class instance.
        period_and_group: Returns the period and group embeddings of the elements.
    """

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
        properties: list = [],
        period: bool = True,
        group: bool = True,
        short: bool = False,
        n_elements: int = 85,
    ) -> None:
        """
        Initializes the PhysRef class.

        Args:
            properties: List of properties to include in the atom
                embeddings. Each property must be a string as per the ``elements`` or
                ``fetch_ionization_energies`` `Mendeleev tables <https://mendeleev.readthedocs.io/en/stable/notebooks/bulk_data_access.html`_.Defaults to [].
            period: Whether to create period mappings, from atomic
                number to period number.
            group: Whether to create group mappings, from atomic
                number to period number.
            short: A boolean flag indicating whether to keep only the
                columns that do not have NaN values.
            n_elements: Number of elements to consider. Defaults to 85.
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
        return f"PhysRef(properties={self.properties}, period={self.period}, group={self.group}, short={self.short})"  # noqa: E501

    def period_and_group(self, z):
        values = {}
        if self.period:
            values["period"] = self.period_mapping[z]
        if self.group:
            values["group"] = self.group_mapping[z]
        return values


class PropertiesEmbedding(nn.Module):
    """
    A class for retrieving physical properties from atomic numbers.

    Args:
        properties (:obj:`torch.Tensor`): A tensor containing the properties to be embedded.
        grad (:obj:`bool`): Whether to enable gradient computation or not.

    Attributes:
        properties (nn.Parameter or nn.Buffer): A parameter or buffer storing the
            properties.

    Methods:
        forward(z): Returns the embedded properties at the specified indices.
        reset_parameters(): Does nothing in this class.
    """

    def __init__(self, properties: torch.Tensor, grad: bool = False):
        """
        Initializes the PropertiesEmbedding object.

        Args:
            properties: A tensor containing the properties to
                use as embeddings.
            grad: Whether properties are fixed or learned (initialized
                from true values then updated according the gradient).
        """
        super().__init__()
        assert isinstance(properties, torch.Tensor)
        assert isinstance(grad, bool)

        if grad:
            self.register_parameter("properties", nn.Parameter(properties))
        else:
            self.register_buffer("properties", properties)

    def forward(self, z: torch.Tensor):
        """
        Returns a properties for each atom in the batch according to
        (1-based) atomic numbers.

        Args:
            z: Tensor of atomic numbers as ``torch.Long``.

        Returns:
            The properties for each atom.
        """
        return self.properties[z]

    def reset_parameters(self):
        pass


class PhysEmbedding(nn.Module):
    """This module embeds inputs for use in a neural network, using both
    standard embeddings and physical properties. The input to the embedding
    module can be a set of compositions, atomic numbers and tags, in addition to
    any extra physical properties specified.

    You can disable embeddings by setting their size to 0.

    Args:
        z_emb_size (:obj:`int`): Size of the embedding for atomic number.
        tag_emb_size (:obj:`int`): Size of the embedding for tags.
        period_emb_size (:obj:`int`): Size of the embedding for periods.
        group_emb_size (:obj:`int`): Size of the embedding for groups.
        properties (:obj:`list`): List of the physical properties to include in the
            embedding. Each property is specified as a string, and should
            correspond to a valid attribute of the Pymatgen Composition
            class.
        properties_proj_size (:obj:`int`): Projection size of the physical properties
            embedding.
        properties_grad (:obj:`bool`): Whether to set the physical properties to be
            trainable or not.
        final_proj_size (:obj:`int`): Projection size for the final embedding.
        n_elements (:obj:`int`): Number of elements in the periodic table.

    Raises:
        ValueError: if `self.properties_proj_size` is greater than 0 and
            `self.properties` is empty
        ValueError: if `self.full_emb_size` is 0, i.e. all sizes were set to 0.

    Attributes:
        z_emb_size (:obj:`int`): Size of the embedding for atomic number.
        tag_emb_size (:obj:`int`): Size of the embedding for tags.
        period_emb_size (:obj:`int`): Size of the embedding for periods.
        group_emb_size (:obj:`int`): Size of the embedding for groups.
        properties (:obj:`list`): List of the physical properties to include in the
            embedding. Each property must be a string as per the ``elements`` or
            ``fetch_ionization_energies`` `Mendeleev tables <https://mendeleev.readthedocs.io/en/stable/notebooks/bulk_data_access.html>`_.
        properties_grad (:obj:`bool`): Whether to set the physical properties to be
            trainable or not.
        n_elements (:obj:`int`): Number of elements in the periodic table to consider.
        phys_ref (PhysRef): Reference physical information interface.
        full_emb_size (:obj:`int`): Total size of the concatenated embeddings.
        final_emb_size (:obj:`int`): Output size: either the final_proj_size or
            full_emb_size.
        embeddings (:obj:`nn.ModuleDict`): Dictionary containing the different
            embeddings.
        phys_lin (:obj:`nn.Linear`): A linear layer to project the physical properties to
            the given size, if projection is requested.
        final_proj (:obj:`nn.Linear`): A linear layer to project the final embedding to
            the requested size.
    """

    def __init__(
        self,
        z_emb_size: int = 32,
        tag_emb_size: int = 0,
        period_emb_size: int = 32,
        group_emb_size: int = 32,
        properties=PhysRef.default_properties,
        properties_grad: bool = False,
        properties_proj_size: int = 0,
        final_proj_size: int = 0,
        n_elements: int = 85,
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
            self.final_emb_size = self.final_proj_size
        else:
            self.final_emb_size = self.full_emb_size

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the linear layers, and the embeddings.
        """
        if self.phys_lin:
            nn.init.xavier_uniform_(self.phys_lin.weight)
        for emb in self.embeddings.values():
            if isinstance(emb, (nn.Sequential, PhysRef)):
                pass
            else:
                emb.reset_parameters()

    def forward(self, z: torch.Tensor, tag: Optional[torch.Tensor] = None):
        """
        Embeds the input(s) using the available embeddings.
        Final embedding size is the sum of the individual embedding sizes,
        except if `final_proj_size` is provided, in which case the final
        embedding is projected to the requested size with an unbiased
        linear layer.

        Args:
            z: Tensor of (long) atomic numbers.
            tag: Open Catalyst Project-style tags. Defaults to None.

        Returns:
            :obj:`torch.Tensor`: Embedded representation of the input(s).
        """
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
