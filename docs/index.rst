.. include:: ../README.rst
    :end-before: when included in index.rst

.. |plot| image:: images/plot_spec.png

How the documentation is structured
-----------------------------------

.. rst-class:: columns

:ref:`tutorials`
~~~~~~~~~~~~~~~~

Tutorials for installation, library and commandline usage. New users start here.

.. rst-class:: columns

:ref:`how-to`
~~~~~~~~~~~~~

Practical step-by-step guides for the more experienced user.

.. rst-class:: columns

:ref:`explanations`
~~~~~~~~~~~~~~~~~~~

Explanation of how the library works and why it works that way.

.. rst-class:: columns

:ref:`reference`
~~~~~~~~~~~~~~~~

Technical reference material, for classes, methods, APIs, commands, and contributing to the project.

.. rst-class:: endcolumns

About the documentation
~~~~~~~~~~~~~~~~~~~~~~~

`Why is the documentation structured this way? <https://documentation.divio.com>`_

.. toctree::
    :caption: Tutorials
    :name: tutorials
    :maxdepth: 1

    tutorials/installation
    tutorials/creating-a-spec
    tutorials/graphql-service

.. toctree::
    :caption: How-to Guides
    :name: how-to
    :maxdepth: 1

    how-to/iterate-a-spec
    how-to/serialize-a-spec

.. toctree::
    :caption: Explanations
    :name: explanations
    :maxdepth: 1

    explanations/technical-terms
    explanations/what-are-dimensions
    explanations/why-squash-can-change-path

.. rst-class:: no-margin-after-ul

.. toctree::
    :caption: Reference
    :name: reference
    :maxdepth: 1

    reference/api
    reference/changelog
    reference/contributing

* :ref:`genindex`
