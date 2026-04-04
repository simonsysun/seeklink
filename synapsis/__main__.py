"""Entry point for `python -m synapsis`."""

import sys

if "--sse" in sys.argv:
    from synapsis.server import main_sse

    main_sse()
else:
    from synapsis.server import main

    main()
