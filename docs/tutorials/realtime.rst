Tutorial 7: Real-Time fNIRS Pipeline
=====================================

Replays Kernel Flow SNIRF data through the dot-jax real-time pipeline
with a web dashboard for live visualization. Designed for the Global
NeuroHack demonstration.

.. literalinclude:: ../../examples/07_realtime_demo.py
   :language: python
   :linenos:

Run it:

.. code-block:: bash

   # Terminal 1: Start the pipeline
   python examples/07_realtime_demo.py

   # Terminal 2: Open browser
   open http://localhost:18080

   # For live acquisition (replace replay with ZMQ receiver):
   python examples/07_realtime_demo.py --live --port 15555
