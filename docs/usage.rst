Usage Guide
===========

Installation
------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/chrisfinlay/tab-sim.git

Install via pip (CPU-only):

.. code-block:: bash

   pip install -e ./tab-sim/

Or with GPU support:

.. code-block:: bash

   pip install -e ./tab-sim/[gpu]

Alternatively, use Docker:

.. code-block:: bash

   docker build -t tab-sim:latest ./tab-sim/
   docker run -it -v $(pwd):/data tab-sim:latest bash

Running Simulations
-------------------

Simulations are defined by YAML config files and can be launched using:

.. code-block:: bash

   sim-vis -c path/to/config.yaml -st spacetrack_login.yaml

For help:

.. code-block:: bash

   sim-vis -h
