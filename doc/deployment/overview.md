# Hannah-Backends

Hannah supports a variety of hardware targets using its configurable backend interface. 

Backends can be enabeld and configured using there respective backend configuration group, e.g.

```sh
    hannah-train backend=<selected_backend>
```

Or by adding the backend to the experiment or global config files. 


## Standalone inference driver

The usual usage of backends is having them integrated into a training or nas run driven by `hannah-train`, but 
it is also possible to run hannah backends in (semi-) standalone mode using command-line driver `hannah-exec`.

It uses a simple command-line driver for backend inference. This means it does not use a simple backend inference driver. 