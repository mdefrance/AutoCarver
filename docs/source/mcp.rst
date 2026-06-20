.. _mcp:

LLM & MCP Integration
=====================

**AutoCarver** can be driven by a Large Language Model to *qualify* a dataset's columns
(decide which are numerical, categorical, ordinal, datetime or nested hierarchies) and then
*carve* them against a target — the two steps that usually require a data scientist's judgment.

There are two ways to use this, depending on where you work:

* :ref:`As an MCP server <mcp_server>`, exposed to an MCP-aware assistant (VS Code Copilot,
  Claude Desktop, Cursor, …) that drives the workflow through tools.
* :ref:`In a notebook <mcp_notebook>`, with an LLM client you instantiate yourself.

Both paths share the **same logic**: the read-only inspection helpers
(:mod:`AutoCarver.mcp.inspection`), the type-routing (:func:`specs_to_features_kwargs`) and the
:class:`~AutoCarver.mcp.session.CarverSession`. The MCP server is only a thin transport layer
on top of that session.

.. note::

    The scope is **qualify + carve**: the workflow stops at the carved-feature artifact (the
    carving summary, per-feature content and a saved carver). It does not train a downstream
    estimator. The saved ``.json`` embeds the carved :class:`Features`, so loading it back with
    :meth:`~AutoCarver.carvers.BinaryCarver.load` restores both the carver and its features.

.. warning::

    **Carving quality depends on the LLM.** Feature *qualification* — deciding which columns are
    ordinal, which form hierarchies and which to ignore (ids, free text, leakage) — is only as
    good as the model you point it at, and different models (or even different runs) can make
    different calls. Treat the LLM's output as a **first draft, not a final answer**: a human
    should review and confirm the feature definitions before any production use.

.. note::

    **Your data stays local.** The MCP server runs entirely on your machine — it reads files
    from your filesystem and never sends your dataset to AutoCarver or any other external
    service. The only data that leaves your machine is whatever your *own* LLM client/provider
    transmits as part of the conversation (e.g. the column summaries the assistant chooses to
    share). That exchange is governed by your provider's terms, not by AutoCarver.



.. _mcp_server:

Path 1 — As an MCP server (VS Code Copilot, Claude Desktop, …)
--------------------------------------------------------------

AutoCarver ships a local `Model Context Protocol <https://modelcontextprotocol.io>`_ server, so
an MCP-aware assistant can build carved features straight from a file path — loading the data,
inspecting it, choosing types and carving, all through tool calls.

Installing the server
^^^^^^^^^^^^^^^^^^^^^

The server depends on ``fastmcp``, which is **not** part of the core install. Pull it in with
the ``mcp`` extra:

.. tab-set::

    .. tab-item:: uv
        :sync: uv

        .. code-block:: bash

            uv add "autocarver[mcp]"

    .. tab-item:: pip
        :sync: pip

        .. code-block:: bash

            pip install "autocarver[mcp]"

Running the server
^^^^^^^^^^^^^^^^^^

The server speaks ``stdio`` (the default transport for MCP clients):

.. code-block:: bash

    python -m AutoCarver.mcp

Clients usually launch the command for you via a config file (below) rather than you running it
by hand.

Configuring the client
^^^^^^^^^^^^^^^^^^^^^^^

Point your client at the command. For **VS Code / GitHub Copilot**, add a ``.vscode/mcp.json``
to your workspace:

.. code-block:: json

    {
      "servers": {
        "autocarver": {
          "command": "uv",
          "args": ["run", "python", "-m", "AutoCarver.mcp"]
        }
      }
    }

For **Claude Desktop** the shape is identical but the top-level key is ``mcpServers`` (in
``claude_desktop_config.json``). 

A typical conversation
^^^^^^^^^^^^^^^^^^^^^^^

Once connected, you can ask the assistant in natural language, e.g.:

    *"Load ``data/loans.parquet`` with target ``default``, profile the columns, qualify the
    feature types, carve them and save the result to ``loans_carver.json``."*

The assistant will chain the tools below: ``load_dataset`` → inspect (``list_columns``,
``feature_distribution``, ``validate_nesting`` …) → draft (``suggest_features`` /
``set_feature``) → ``run_carver`` → ``save_carver``, and report the carving summary back to you.

Available tools
^^^^^^^^^^^^^^^

**Inspection (read-only).** Return summaries only — never raw columns wholesale — so they stay
safe within a model's context window.

* ``load_dataset(path, target=None)`` — load a ``.csv`` / ``.parquet`` as the working dataset.
* ``list_columns()`` — dtype, cardinality, missingness and suggested kind for every column.
* ``profile_column(column, top_n=20)`` — quantiles (numeric) or top modalities (qualitative).
* ``feature_distribution(column, min_freq=None, top_n=50)`` — modality counts, target rate and
  rare-modality flags (Wilson confidence interval, matching how the carvers judge rarity).
* ``validate_nesting(child, parents)`` — check a column rolls cleanly (many-to-one) into coarser
  parents before declaring a ``nested`` feature.
* ``datetime_reference_candidates()`` — span and coverage of datetime columns to pick a reference.

**Drafting & carving.**

* ``suggest_features()`` — fill the draft with dtype-based suggestions (skips the target).
* ``set_feature(column, kind, values=None, reference=None, parents=None)`` — set/override one
  column's kind (``numerical`` / ``categorical`` / ``ordinal`` / ``datetime`` / ``nested`` / ``ignore``).
* ``drop_feature(column)`` — remove a column from the draft.
* ``preview_features()`` — return the current draft as ``{column: spec}``.
* ``run_carver(task="auto", min_freq=0.05, max_n_mod=5)`` — build ``Features`` from the draft
  and carve them against the target; returns the kept/dropped features, carved content and summary.
* ``save_carver(path)`` — save the fitted carver and its carved features to a ``.json`` file
  (run ``run_carver`` first); the file restores both via ``BinaryCarver.load``.



.. _mcp_notebook:

Path 2 — In a notebook, with your own LLM
-----------------------------------------

You already have a dataset loaded and an LLM client instantiated. AutoCarver gives you a
provider-agnostic qualifier: you supply ``llm_fn``, a callable that takes a prompt string and
returns the model's raw text. No provider SDK is imported by AutoCarver, so any backend works.

Qualifying columns into ``Features``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from anthropic import Anthropic
    from AutoCarver.features import qualify_with_llm

    client = Anthropic()

    def llm_fn(prompt: str) -> str:
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    # X is your loaded DataFrame; the model qualifies every column
    features = qualify_with_llm(X, llm_fn)

:func:`~AutoCarver.features.qualify_with_llm` builds a prompt describing each column (dtype,
cardinality, a value sample), asks the model to return a JSON ``{column: spec}`` mapping, and
routes it into a :class:`~AutoCarver.features.Features`. Any backend works — swap ``llm_fn`` for
an OpenAI / local-model call.

Then carve exactly as you would by hand:

.. code-block:: python

    from AutoCarver import BinaryCarver

    carver = BinaryCarver(features=features, min_freq=0.02, max_n_mod=5)
    x_discretized = carver.fit_transform(X, X[target])
    carver.save("my_carver.json")

.. note::

    Without an LLM, :meth:`Features.from_dataframe` gives the same dtype-based first pass
    deterministically. ``qualify_with_llm`` adds the semantic judgment a dtype cannot infer:
    *ordering* (ordinals), *hierarchies* (nested) and *which columns to ignore* (ids, free
    text, leakage).

Driving the session programmatically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a guided, tool-by-tool workflow (the same one the MCP server exposes) without running a
server, use :class:`~AutoCarver.mcp.session.CarverSession` directly. This lets the model — or
you — *interrogate* the data before committing to types:

.. code-block:: python

    from AutoCarver.mcp import CarverSession

    session = CarverSession()
    session.load_dataset("data.parquet", target="default")

    session.list_columns()                         # dtype / cardinality / missingness / suggested kind
    session.feature_distribution("job", min_freq=0.05)   # modality rates + rare-modality flags
    session.validate_nesting("city", ["region", "country"])  # check a hierarchy is many-to-one

    session.suggest_features()                      # dtype-based draft (skips the target)
    session.set_feature("grade", "ordinal", values=["low", "medium", "high"])
    result = session.run_carver(task="auto", min_freq=0.05, max_n_mod=5)
    session.save_carver("my_carver.json")          # carver + carved features, reloadable

``run_carver`` returns the resolved task, the kept/dropped features, the carved content per
feature and the carving summary. ``save_carver`` persists the fitted carver — features
included — so it can later be restored with ``BinaryCarver.load("my_carver.json")``.



Architecture
------------

The MCP package is layered so the logic is importable and unit-tested without ``fastmcp``:

* :mod:`AutoCarver.mcp.inspection` — pure, read-only functions over a ``DataFrame``.
* :mod:`AutoCarver.mcp.session` — :class:`CarverSession`, the stateful ``load → draft → carve``
  workflow; each method maps one-to-one to a tool.
* :mod:`AutoCarver.mcp.server` — :func:`create_server` (FastMCP), a thin wrapper registering
  each session method as a tool.

The draft specs use the **same schema** as :func:`~AutoCarver.features.qualify_with_llm`, and
both go through :func:`~AutoCarver.features.specs_to_features_kwargs` — a single source of truth
for routing a column's declared type into a :class:`~AutoCarver.features.Features`.
