Installation
============

**AutoCarver** can be installed from `PyPI <https://pypi.org/project/AutoCarver>`_.

Pick your package manager below — the choice applies to every command on this page.

Light install (recommended for production)
------------------------------------------

To install core features, use the following:

.. tab-set::

    .. tab-item:: uv
        :sync: uv

        .. code-block:: bash

            uv add autocarver

    .. tab-item:: pip
        :sync: pip

        .. code-block:: bash

            pip install autocarver

Driving AutoCarver from an LLM (MCP server)
-------------------------------------------

To run the local MCP server (so an MCP-aware assistant can qualify and carve features through
tool calls), install the ``mcp`` extra:

.. tab-set::

    .. tab-item:: uv
        :sync: uv

        .. code-block:: bash

            uv add "autocarver[mcp]"

    .. tab-item:: pip
        :sync: pip

        .. code-block:: bash

            pip install "autocarver[mcp]"

See :ref:`mcp` for client configuration and the available tools.

Enabling pretty printing (recommended for development)
------------------------------------------------------

To enable HTML outputs (nice colorful tables within jupyter), use the following:

.. tab-set::

    .. tab-item:: uv
        :sync: uv

        .. code-block:: bash

            uv add "autocarver[jupyter]"

    .. tab-item:: pip
        :sync: pip

        .. code-block:: bash

            pip install "autocarver[jupyter]"
