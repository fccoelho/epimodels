"""
VFGen XML exporter for epimodels.

This module provides functionality to export ContinuousModel instances
to the vfgen XML format, enabling use with various numerical analysis tools.

VFGen XML Specification: https://warrenweckesser.github.io/vfgen/menu_fileformat.html
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.dom import minidom

if TYPE_CHECKING:
    import sympy as sp
    from epimodels.continuous import ContinuousModel

from epimodels.formulas import (
    FormulaExtractionError,
    extract_formulas,
    get_free_symbols,
    sympy_to_vfgen,
    validate_formulas,
)


class VFGenExporter:
    """
    Export epimodels ContinuousModel instances to vfgen XML format.

    VFGen is a tool that converts vector field definitions to various
    formats for numerical analysis (AUTO, MATLAB, Scipy, etc.).

    Attributes:
        model: The ContinuousModel instance to export

    Example:
        >>> from epimodels.continuous import SIR
        >>> from epimodels.exporters import VFGenExporter
        >>> model = SIR()
        >>> model.param_values = {'beta': 0.3, 'gamma': 0.1}
        >>> exporter = VFGenExporter(model)
        >>> xml = exporter.export(initial_conditions={'S': 990, 'I': 10, 'R': 0})
    """

    def __init__(self, model: ContinuousModel):
        """
        Initialize the exporter.

        Args:
            model: A ContinuousModel instance

        Raises:
            TypeError: If model is not a ContinuousModel
        """
        # Check if model has required attributes
        if not hasattr(model, "_model"):
            raise TypeError(
                f"{type(model).__name__} does not appear to be a ContinuousModel. "
                f"VFGen export only supports continuous-time models with a _model method."
            )

        if not hasattr(model, "state_variables"):
            raise TypeError(
                f"{type(model).__name__} does not have state_variables attribute. "
                f"Ensure the model is a properly defined ContinuousModel."
            )

        self.model = model

    def export(
        self,
        filepath: str | Path | None = None,
        default_values: dict[str, float] | None = None,
        initial_conditions: dict[str, float] | None = None,
        population: float | None = None,
        include_description: bool = True,
        include_latex: bool = True,
        include_n_constant: bool = True,
        expressions: dict[str, sp.Expr] | None = None,
        functions: dict[str, sp.Expr] | None = None,
        validate_formulas_flag: bool = True,
    ) -> str | None:
        """
        Export the model to vfgen XML format.

        Args:
            filepath: Optional path to write XML file. If None, returns XML string.
            default_values: Parameter default values. Uses model.param_values if None.
            initial_conditions: Initial conditions for state variables.
            population: Total population N value. Required if include_n_constant is True.
            include_description: Include Description attributes in XML.
            include_latex: Include Latex attributes in XML.
            include_n_constant: Include N as a Constant element.
            expressions: Additional Expression elements (name -> formula).
            functions: Function elements (name -> formula).
            validate_formulas_flag: Whether to validate formulas before export.

        Returns:
            XML string if filepath is None, otherwise None (writes to file)

        Raises:
            FormulaExtractionError: If formula extraction fails
            ValueError: If required parameters are missing
        """
        # Get formulas (tries automatic extraction first, then manual override)
        if validate_formulas_flag:
            formulas = self.model.get_formulas()
        else:
            formulas = getattr(self.model, "_formulas", None)
            if formulas is None:
                formulas = extract_formulas(self.model)

        if formulas is None:
            raise FormulaExtractionError(
                model_name=getattr(self.model, "model_type", type(self.model).__name__),
                reason="Could not obtain formulas for the model",
                suggestion="Define _formulas manually or ensure _model method is compatible.",
            )

        # Build XML
        root = self._build_xml_root()

        # Add parameters
        self._add_parameters(
            root,
            default_values=default_values,
            include_description=include_description,
            include_latex=include_latex,
        )

        # Add N constant if requested
        if include_n_constant:
            if population is not None:
                self._add_constant(root, "N", population)

        # Add expressions
        if expressions:
            for name, expr in expressions.items():
                formula_str = sympy_to_vfgen(expr)
                self._add_expression(root, name, formula_str)

        # Add state variables
        self._add_state_variables(
            root,
            formulas=formulas,
            initial_conditions=initial_conditions,
            include_description=include_description,
            include_latex=include_latex,
        )

        # Add functions
        if functions:
            for name, expr in functions.items():
                formula_str = sympy_to_vfgen(expr)
                self._add_function(root, name, formula_str)

        # Generate XML string
        xml_string = self._prettify(root)

        # Write to file or return string
        if filepath is not None:
            filepath = Path(filepath)
            filepath.write_text(xml_string)
            return None

        return xml_string

    def _build_xml_root(self) -> ET.Element:
        """Create the root VectorField element."""
        root = ET.Element("VectorField")

        # Set Name attribute (required)
        model_name = getattr(self.model, "model_type", type(self.model).__name__)
        root.set("Name", model_name)

        # Set optional attributes
        if hasattr(self.model, "name") and self.model.name:
            root.set("Name", self.model.name)

        return root

    def _add_parameters(
        self,
        root: ET.Element,
        default_values: dict[str, float] | None,
        include_description: bool,
        include_latex: bool,
    ) -> None:
        """Add Parameter elements to the XML."""
        defaults: dict[str, float] = default_values or getattr(self.model, "param_values", {}) or {}

        for param_name in self.model.parameters:
            elem = ET.SubElement(root, "Parameter")
            elem.set("Name", param_name)

            # Add default value if available
            if param_name in defaults:
                elem.set("DefaultValue", str(defaults[param_name]))

            # Add description from state_variables dict values
            if include_description:
                description = self.model.parameters.get(param_name, "")
                if description:
                    # Strip LaTeX markers for description
                    clean_desc = description.replace("$", "").replace("\\", "")
                    elem.set("Description", clean_desc)

            # Add LaTeX symbol
            if include_latex:
                latex = self.model.parameters.get(param_name)
                if latex:
                    elem.set("Latex", latex)

    def _add_constant(
        self,
        root: ET.Element,
        name: str,
        value: float,
        description: str = "",
    ) -> None:
        """Add a Constant element to the XML."""
        elem = ET.SubElement(root, "Constant")
        elem.set("Name", name)
        elem.set("Value", str(value))
        if description:
            elem.set("Description", description)

    def _add_expression(
        self,
        root: ET.Element,
        name: str,
        formula: str,
        description: str = "",
    ) -> None:
        """Add an Expression element to the XML."""
        elem = ET.SubElement(root, "Expression")
        elem.set("Name", name)
        elem.set("Formula", formula)
        if description:
            elem.set("Description", description)

    def _add_state_variables(
        self,
        root: ET.Element,
        formulas: dict[str, sp.Expr],
        initial_conditions: dict[str, float] | None,
        include_description: bool,
        include_latex: bool,
    ) -> None:
        """Add StateVariable elements to the XML."""
        if initial_conditions is None:
            initial_conditions = {}

        for state_name in self.model.state_variables:
            elem = ET.SubElement(root, "StateVariable")
            elem.set("Name", state_name)

            # Add formula (required)
            if state_name in formulas:
                formula_str = sympy_to_vfgen(formulas[state_name])
                elem.set("Formula", formula_str)

            # Add initial condition if available
            if state_name in initial_conditions:
                elem.set("DefaultInitialCondition", str(initial_conditions[state_name]))

            # Add description
            if include_description:
                description = self.model.state_variables.get(state_name, "")
                if description:
                    elem.set("Description", description)

            # Add LaTeX symbol (usually just the variable name)
            if include_latex:
                elem.set("Latex", state_name)

    def _add_function(
        self,
        root: ET.Element,
        name: str,
        formula: str,
        description: str = "",
    ) -> None:
        """Add a Function element to the XML."""
        elem = ET.SubElement(root, "Function")
        elem.set("Name", name)
        elem.set("Formula", formula)
        if description:
            elem.set("Description", description)

    def _prettify(self, root: ET.Element) -> str:
        """
        Return a prettified XML string.

        Args:
            root: The root XML element

        Returns:
            Formatted XML string with declaration
        """
        # Convert to string
        rough_string = ET.tostring(root, encoding="unicode")

        # Parse with minidom for pretty printing
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split("\n") if line.strip()]

        # Ensure proper XML declaration
        if not lines[0].startswith("<?xml"):
            lines.insert(0, '<?xml version="1.0" ?>')
        elif lines[0].startswith('<?xml version="1.0"?>'):
            # Add space before ?>
            lines[0] = '<?xml version="1.0" ?>'

        return "\n".join(lines)


def export_to_vfgen(
    model: ContinuousModel,
    filepath: str | Path | None = None,
    **kwargs,
) -> str | None:
    """
    Convenience function to export a model to vfgen XML.

    Args:
        model: A ContinuousModel instance
        filepath: Optional output file path
        **kwargs: Additional arguments passed to VFGenExporter.export()

    Returns:
        XML string if filepath is None, otherwise None
    """
    return VFGenExporter(model).export(filepath=filepath, **kwargs)
