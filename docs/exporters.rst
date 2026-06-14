VFGen XML Exporter
===================

Epimodels can export continuous-time (ODE) models to the **VFGen XML**
format, enabling use with external numerical analysis tools such as
AUTO, MATLAB, and Scipy-based utilities.

VFGen is a tool that converts vector field definitions into code for
various target languages and analysis frameworks. See the
`VFGen documentation <https://warrenweckesser.github.io/vfgen/menu_fileformat.html>`_
for the full XML specification.

.. contents::
   :local:
   :depth: 1

Quick Start
-----------

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.exporters import VFGenExporter

    model = SIR()
    model.param_values = {'beta': 0.3, 'gamma': 0.1}

    exporter = VFGenExporter(model)

    # Export to file
    exporter.export("sir_model.xml", initial_conditions={'S': 990, 'I': 10, 'R': 0})

    # Export to string
    xml_str = exporter.export(
        initial_conditions={'S': 990, 'I': 10, 'R': 0},
        population=1000,
    )
    print(xml_str)

The :class:`~epimodels.exporters.VFGenExporter` automatically extracts
the model's ODE formulas via symbolic analysis and converts them to the
VFGen XML schema.

Export Options
--------------

The :meth:`~epimodels.exporters.VFGenExporter.export` method supports
several options:

.. code-block:: python

    exporter = VFGenExporter(model)

    # Full export with all options
    xml = exporter.export(
        filepath="model.xml",                  # Write to file (None = return string)
        default_values={'beta': 0.3, 'gamma': 0.1},  # Override parameter defaults
        initial_conditions={'S': 990, 'I': 10, 'R': 0},
        population=1000,                       # Total population (N)
        include_description=True,              # Include Description attributes
        include_latex=True,                    # Include LaTeX symbol attributes
        include_n_constant=True,               # Include N as a Constant element
        validate_formulas_flag=True,           # Validate formulas before export
    )

Advanced: Custom Expressions and Functions
------------------------------------------

You can add custom expressions and functions to the XML output:

.. code-block:: python

    import sympy as sp

    # Custom expression (e.g., force of infection)
    beta, S, I, N = sp.symbols('beta S I N')
    foi = beta * S * I / N

    exporter = VFGenExporter(model)
    xml = exporter.export(
        expressions={'FOI': foi},              # Additional Expression elements
        functions={'log_base2': sp.log(2)},    # Additional Function elements
    )

Supported Models
----------------

VFGen export is supported for all continuous-time models that have a
well-defined :meth:`_model` method, including:

- :class:`~epimodels.continuous.models.SIR`
- :class:`~epimodels.continuous.models.SIS`
- :class:`~epimodels.continuous.models.SIRS`
- :class:`~epimodels.continuous.models.SEIR`
- :class:`~epimodels.continuous.models.SEQIAHR`
- :class:`~epimodels.continuous.models.SIRSEI`
- :class:`~epimodels.continuous.models.SIR2Strain`
- All other :class:`~epimodels.continuous.models.ContinuousModel` subclasses

If automatic formula extraction fails for a custom model, you can define
``_formulas`` manually on the model class.

API Reference
-------------

.. autoclass:: epimodels.exporters.VFGenExporter
   :members:
   :undoc-members:
